use std::f64::consts::PI;

use candle_nn::{AdamW, Optimizer};

#[derive(Debug)]
pub struct OneCycle {
    lr: f64,
    momentum: f64,
    step_num: usize,
    phases: Vec<Phase>,
}

#[derive(Debug)]
pub struct Phase {
    end_step: usize,
    start_lr: f64,
    end_lr: f64,
    start_momentum: f64,
    end_momentum: f64,
}

fn cos_annealing(start: f64, end: f64, pct: f64) -> f64 {
    let cos_out = (pct * PI).cos() + 1.;
    end + (start - end) / 2. * cos_out
}

fn build_phases(
    max_lr: f64,
    min_lr: f64,
    max_momentum: f64,
    min_momentum: f64,
    total_steps: usize,
    percent_start: f64,
) -> Vec<Phase> {
    vec![
        Phase {
            end_step: (percent_start * total_steps as f64 - 1.).round() as usize,
            start_lr: min_lr,
            end_lr: max_lr,
            start_momentum: max_momentum,
            end_momentum: min_momentum,
        },
        Phase {
            end_step: total_steps - 1,
            start_lr: max_lr,
            end_lr: min_lr,
            start_momentum: min_momentum,
            end_momentum: max_momentum,
        },
    ]
}

impl OneCycle {
    pub fn new(max_lr: f64, max_momentum: f64, div_factor: f32, total_steps: usize) -> Self {
        let min_lr = max_lr / div_factor as f64;

        OneCycle {
            lr: min_lr,
            momentum: max_momentum,
            phases: build_phases(
                max_lr,
                max_lr / div_factor as f64,
                max_momentum,
                max_momentum / div_factor as f64,
                total_steps,
                0.3,
            ),
            step_num: 0,
        }
    }

    pub fn step(&mut self, optimizer: &mut AdamW) {
        self.step_num += 1;

        let mut start_step = 0;

        for phase in self.phases.as_slice() {
            if self.step_num <= phase.end_step {
                let pct =
                    (self.step_num - start_step) as f64 / (phase.end_step - start_step) as f64;
                self.lr = cos_annealing(phase.start_lr, phase.end_lr, pct);

                self.momentum = cos_annealing(phase.start_momentum, phase.end_momentum, pct);
                break;
            };
            start_step = phase.end_step;
        }

        optimizer.set_learning_rate(self.lr);
        let mut params = optimizer.params().clone();
        params.beta1 = self.momentum;
        optimizer.set_params(params.clone());
    }

    pub fn get_lr(&self) -> f64 {
        self.lr
    }

    pub fn get_momentum(&self) -> f64 {
        self.momentum
    }
}

#[derive(Debug)]
pub struct CosineAnnealing {
    base_lr: f64,
    lr: f64,
    eta_min: f64,
    max_step: usize,
    step_num: usize,
}

impl CosineAnnealing {
    pub fn new(lr: f64, max_step: usize, eta_min: f64) -> Self {
        CosineAnnealing {
            base_lr: lr,
            lr,
            eta_min,
            max_step,
            step_num: 0,
        }
    }

    pub fn step(&mut self, optimizer: &mut AdamW) {
        self.step_num += 1;

        self.lr = self.eta_min
            + (self.base_lr - self.eta_min)
                * (1. + (PI * self.step_num as f64 / self.max_step as f64).cos())
                / 2.;

        optimizer.set_learning_rate(self.lr);
    }

    pub fn get_lr(&self) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

    use crate::{CosineAnnealing, OneCycle};

    #[test]
    fn one_cycle_test() {
        let varmap = VarMap::new();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = OneCycle::new(1e-3, 0.9, 25., 10);

        scheduler.step(&mut opt);

        assert_eq!(scheduler.get_lr(), 0.0005200000000000001);
        assert_eq!(scheduler.get_momentum(), 0.46799999999999997);
    }

    #[test]
    fn one_cycle_mid_test() {
        let varmap = VarMap::new();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = OneCycle::new(1e-3, 0.9, 25., 10);

        // Go to mid
        for _i in 0..=5 {
            scheduler.step(&mut opt);
        }

        assert_eq!(scheduler.get_lr(), 0.0004131899517009691);
        assert_eq!(scheduler.get_momentum(), 0.5641290434691278);
    }

    #[test]
    fn one_cycle_end_test() {
        let varmap = VarMap::new();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = OneCycle::new(1e-3, 0.9, 25., 10);

        // Go to mid
        for _i in 0..=10 {
            scheduler.step(&mut opt);
        }

        assert_eq!(scheduler.get_lr(), 4e-5);
        assert_eq!(scheduler.get_momentum(), 0.9);
    }

    #[test]
    fn cosine_annealing_test() {
        let varmap = VarMap::new();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = CosineAnnealing::new(1e-3, 10, 1e-6);

        scheduler.step(&mut opt);

        assert_eq!(scheduler.get_lr(), 0.0009755527298894294);
    }

    #[test]
    fn cosine_annealing_mid_test() {
        let varmap = VarMap::new();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = CosineAnnealing::new(1e-3, 10, 1e-6);

        for _i in 0..=5 {
            scheduler.step(&mut opt);
            println!("{}", scheduler.get_lr());
        }

        assert_eq!(scheduler.get_lr(), 0.0003461460113097139);
    }

    #[test]
    fn cosine_annealing_end_test() {
        let varmap = VarMap::new();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = CosineAnnealing::new(1e-3, 10, 1e-6);

        for _i in 0..=10 {
            scheduler.step(&mut opt);
        }

        assert_eq!(scheduler.get_lr(), 2.5447270110570702e-5);
    }
}
