import os
import json
import datetime
import functools
import numpy as np

import utils as utils
from settings import settings, RegressionSettings

optimization_config = settings.optimization_config

def opro():
  pass

def main(*args, **kwargs):
  w_true = 36
  b_true = -1

  # load LLM settings
  optimizer_llm_name = optimization_config["model"]
  reg = RegressionSettings(w_true=w_true, b_true=b_true, num_points=100)

  # create the result directory
  datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
  )

  save_folder = os.path.join(
      settings.ROOT_PATH,
      "outputs",
      "optimization-results",
      f"linear_regression-{optimizer_llm_name}-w_{w_true}-b_{b_true}-{datetime_str}/",
  )
  os.makedirs(save_folder)
  print(f"result directory:\n{save_folder}")

  # ====================== optimizer model configs ============================

  call_gemini_api_func = functools.partial(
      utils.call_gemini_api,
      model=optimizer_llm_name,
      temperature=optimization_config["temperature"],
      max_decode_steps=optimization_config["max_decode_steps"],
  )

  optimizer_llm_dict = {
      "model_type": optimizer_llm_name.lower(),
  }
  optimizer_llm_dict.update(optimization_config)
  call_optimizer_server_func = call_gemini_api_func

  configs_dict = dict()
  results_dict = dict()
  num_convergence_steps = []
  for i_rep in range(reg.num_reps):
    found_optimal = False
    print(f"\nRep {i_rep}:")

    X, y = reg.get_training_data()
    loss_at_true_values = RegressionSettings.evaluate_loss(X, y, reg.w_true, reg.b_true)
    print(f"value at (w_true, b_true): {loss_at_true_values}")

    # ================= generate the starting points =====================
    old_value_pairs_set = set()
    old_value_pairs_with_i_step = []  # format: [(w, b, z = f(w, b), i_step)]
    meta_prompts_dict = dict()  # format: {i_step: meta_prompt}
    raw_outputs_dict = dict()  # format: {i_step: raw_outputs}

    params = reg.init_params()
    w,b,z = list(zip(*params))


    for weight, bias, score in params:
        old_value_pairs_set.add((weight, bias, score))
        old_value_pairs_with_i_step.append((weight, bias, score, -1))


    # ====================== run optimization ============================
    configs_dict_single_rep = {
        "optimizer_llm_configs": optimizer_llm_dict,
        "data": {
            "num_points": reg.num_points,
            "w_true": reg.w_true,
            "b_true": reg.b_true,
            "loss_at_true_values": loss_at_true_values,
            "X": list(X),
            "y": list(y),
        },
        "init_w": list(w),
        "init_b": list(b),
        "max_num_steps": reg.max_steps,
        "max_num_pairs": reg.max_num_pairs,
        "num_input_decimals": reg.num_input_decimals,
        "num_output_decimals": reg.num_output_decimals,
        "num_generated_points_in_each_step": reg.num_generated_points_in_each_step,
    }
    configs_dict[i_rep] = configs_dict_single_rep
    configs_json_path = os.path.join(save_folder, "configs.json")
    print(f"saving configs to\n{configs_json_path}")
    with open(configs_json_path, "w") as f:
      json.dump(configs_dict, f, indent=4)


    print("\n================ run optimization ==============")
    print(f"initial values: {[item[-1] for item in old_value_pairs_set]}")
    results_json_path = os.path.join(save_folder, "results.json")
    print(f"saving results to\n{results_json_path}")

    for i_step in range(reg.max_steps):
      print(f"\nStep {i_step}:")
      meta_prompt = utils.gen_meta_prompt(
          old_value_pairs_set,
          X,
          y,
          num_input_decimals=reg.num_input_decimals,
          num_output_decimals=reg.num_output_decimals,
          max_num_pairs=reg.max_num_pairs,
      )
      if not i_step % 5:
        print("\n=================================================")
        print(f"meta_prompt:\n{meta_prompt}")
      meta_prompts_dict[i_step] = meta_prompt

      # generate a maximum of the given number of points in each step
      remaining_num_points_to_generate = reg.num_generated_points_in_each_step
      raw_outputs = []
      while remaining_num_points_to_generate > 0:
        raw_outputs.append(call_optimizer_server_func(meta_prompt))
        remaining_num_points_to_generate -= optimizer_llm_dict["batch_size"]
      raw_outputs = raw_outputs[:reg.num_generated_points_in_each_step]

      raw_outputs_dict[i_step] = raw_outputs
      parsed_outputs = []
      for output in raw_outputs:
        try:
          json_output = json.loads(output)
        except json.JSONDecodeError:
          continue
        try:
          parsed_output = json_output["weight_bias_pair"]
          if parsed_output is not None and len(parsed_output) == 2:
            parsed_outputs.append(parsed_output)
        except ValueError:
          pass
      print(f"proposed points before rounding: {parsed_outputs}")

      # round the proposed points to the number of decimals in meta-prompt
      rounded_outputs = [
          (np.round(w, reg.num_input_decimals), np.round(b, reg.num_input_decimals))
          for w, b in parsed_outputs
      ]
      rounded_outputs = [
          tuple(item) for item in list(np.unique(rounded_outputs, axis=0))
      ]
      print(f"proposed points after rounding: {rounded_outputs}")

      # evaluate the values of proposed and rounded outputs
      single_step_values = []
      for w, b in rounded_outputs:
        if w == reg.w_true and b == reg.b_true:
          found_optimal = True
        z = RegressionSettings.evaluate_loss(X, y, w, b)
        single_step_values.append(z)
        old_value_pairs_set.add((w, b, z))
        old_value_pairs_with_i_step.append((w, b, z, i_step))
      print(f"single_step_values: {single_step_values}")

      # ====================== save results ============================
      results_dict_single_rep = {
          "meta_prompts": meta_prompts_dict,
          "raw_outputs": raw_outputs_dict,
          "old_value_pairs_with_i_step": old_value_pairs_with_i_step,
      }
      results_dict[i_rep] = results_dict_single_rep
      with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=4, cls=utils.NpEncoder)
      if found_optimal:
        print(
            f"Repetition {i_rep+1}, optimal found at Step {i_step+1}, saving"
            f" final results to\n{save_folder}"
        )
        num_convergence_steps.append(i_step + 1)
        break
  print(f"num_convergence_steps: {num_convergence_steps}")


if __name__ == "__main__":
  main()
