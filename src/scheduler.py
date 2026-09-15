class NoamScheduler:
    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def get_lr(self, step: int) ->float:
        if step <= 0:
            return 0.0
        return (
            self.factor
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )

    def step(self):
        self.step_num += 1
        lr = self.get_lr(self.step_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self):
        return {
            "step_num": self.step_num
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]
        lr = self.get_lr(self.step_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr