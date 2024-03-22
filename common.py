from typing import OrderedDict


def manipulate_state_dict(state_dict: OrderedDict, args: dict):
    """manipulate the state dict to make it loadable of the MoE model.

    Args:
        state_dict (OrderedDict): The state
        args (dict): TODO

    Returns: TODO

    """
    new_state_dict = {}
    if args.adapter_type == "smear":
        num_experts = args.adapter_config["num_experts"]
        if "deberta.embeddings.linear_video.router.temperature" in state_dict:
            return state_dict  # finetuned checkpoint.

        for k, v in state_dict.items():
            if "adapter" in k or "linear_video" in k:  # adapters
                v_ndim = v.ndim
                v = v.unsqueeze(0).repeat(num_experts, *([1] * v_ndim))
                if args.add_noise_to_expert > 0:
                    noise = torch.empty(v.shape)
                    torch.nn.init.normal_(noise, mean=0, std=args.add_noise_to_expert)
                    v = v + noise
                if "linear_video" in k:
                    k = k.replace("linear_video", "linear_video.linear")
            new_state_dict[k] = v

    elif args.adapter_type == "smear_cl":
        num_experts = args.adapter_config["num_experts"]
        if "deberta.embeddings.linear_video.router.temperature" in state_dict:
            return state_dict  # finetuned checkpoint.
        for k, v in state_dict.items():
            if "adapter" in k:
                for i in range(num_experts):
                    new_state_dict[k + f".{i}"] = v
            elif "linear_video" in k:
                for i in range(num_experts):
                    new_state_dict[
                        k.replace("linear_video", "linear_video.linear") + f".{i}"
                    ] = v
            else:
                new_state_dict[k] = v
    elif args.adapter_type == "smear_cl_v2":
        if "deberta.embeddings.linear_video.linear.weight" in state_dict:
            return state_dict  # finetuned checkpoint.
        for k, v in state_dict.items():
            if "linear_video" in k:
                new_state_dict[k.replace("linear_video", "linear_video.linear")] = v
            else:
                new_state_dict[k] = v
    else:
        new_state_dict = state_dict
    return new_state_dict
