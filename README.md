# candle_scheduler

[Candle](https://github.com/huggingface/candle) scheduler

- OneCycleLR

```rust
let params = ParamsAdamW {
    // LR here will be set by the scheduler
    ..Default::default()
};

let mut opt = AdamW::new(varmap.all_vars(), params)?;

// Set the LR and the total number of steps
let mut scheduler = scheduler::OneCycleLR::new(1e-2, total_steps, 25.);

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
}

```
