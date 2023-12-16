use std::f64::consts::PI;

use candle_nn::{AdamW, Optimizer};

#[derive(Debug)]
pub struct OneCycleLR {
    // max_lr: f64,
    // min_lr: f64,
    lr: f64,
    // total_steps: usize,
    // div_factor: f32,
    // final_div_factor: f32,
    step_num: usize,
    phases: Vec<LRPhase>,
}

#[derive(Debug)]
pub struct LRPhase {
    end_step: usize,
    start_lr: f64,
    end_lr: f64,
}

fn cos_annealning(start: f64, end: f64, pct: f64) -> f64 {
    let cos_out = (pct * PI).cos() + 1.;
    end + (start - end) / 2. * cos_out
}

impl OneCycleLR {
    pub fn new(max_lr: f64, total_steps: usize, div_factor: f32) -> Self {
        let min_lr = max_lr / div_factor as f64;

        let phases = vec![
            LRPhase {
                end_step: (0.3 * total_steps as f64 - 1.).round() as usize,
                start_lr: min_lr,
                end_lr: max_lr,
            },
            LRPhase {
                end_step: total_steps - 1,
                start_lr: max_lr,
                end_lr: min_lr,
            },
        ];

        OneCycleLR {
            lr: min_lr,
            phases,
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
                self.lr = cos_annealning(phase.start_lr, phase.end_lr, pct);
                break;
            };
            start_step = phase.end_step;
        }

        optimizer.set_learning_rate(self.lr);
    }

    pub fn get_lr(&self) -> f64 {
        self.lr
    }
}
