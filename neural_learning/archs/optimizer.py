def set_optimizers(self):
    model_dim = self.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()
    # Separate out all parameters into groups
    matrix_params = list(self.transformer.h.parameters())
    value_embeds_params = list(self.value_embeds.parameters())
    embedding_params = list(self.transformer.wte.parameters())
    lm_head_params = list(self.lm_head.parameters())
    resid_params = [self.resid_lambdas]
    x0_params = [self.x0_lambdas]
    assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + ...
    len(value_embeds_params) + len(resid_params) + len(x0_params)
    
    # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
    # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale),  # same LR as token embedding
        dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
        dict(params=x0_params, lr=scalar_lr, betas=(0.96, 0.95)), # higher beta1 for x0 scalars
    ]
    adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
    AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
    adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
    # Create the Muon optimizer for the linear layers
    muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
    MuonFactory = DistMuon if ddp else Muon
    muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
    # Combine them the two optimizers into one list
    optimizers = [adamw_optimizer, muon_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers