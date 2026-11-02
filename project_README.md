What i want to do:
- modify the gpt architecture and different data mix (fewer params, more data)
- see if can come close to the GPT-2 val bpb (change it. for a fixed compute, like number of FLOPS (eg 4xH100s for exactly 2 hours), see what changing the architecture of GPT does to the val loss as i change the architecture, RoPE, GQA to diff degrees, want to see if training on smaller model for lesser tokens can also )
- see if GQA speeds up inference?
- see how gqa of different ratios affects training
- explore if GQA will degrade performance
- Does partial RoPE reduce training time?
- use a dataloader instead for streaming?
- See if GQA improves the inference TPS on same compute (like one GPU).  