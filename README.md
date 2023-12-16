# candle_scheduler

[Candle](https://github.com/huggingface/candle) scheduler

- OneCycleLR

```rust
let varmap = VarMap::new();

let params = ParamsAdamW {
    // LR here will be set by the scheduler
    ..Default::default()
};

let mut opt = AdamW::new(varmap.all_vars(), params)?;

// total number of steps
let total_steps = 10;

// The div factor for the minimum Learning Rate (LR)
let div_factor = 25.; 

// Set the Max LR
let max_lr = 1e-2;
let mut scheduler = scheduler::OneCycleLR::new(max_lr, total_steps, div_factor);

// Learning steps
for i in 0..10 {
    // Some logits from the model
    let logits = Tensor::rand(...);

    // Calculate the loss against the target
    let loss = loss::mse(&logits, targets);

    // Backwards pass
    opt.backward_step(&loss)?;

    // Then we update the LR with the scheduler
    scheduler.step(&mut opt);

    println!("{}", scheduler.get_lr());
}
```
