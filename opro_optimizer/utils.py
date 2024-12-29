import time
import json
import torch
from typing import List
import numpy as np
import google.generativeai as genai
import typing_extensions as typing

class LinearRegressionSchema(typing.TypedDict):
  reasoning: str
  weight_bias_pair: List[int]

class NpEncoder(json.JSONEncoder):
  def default(self, obj):
      if isinstance(obj, np.integer):
          return int(obj)
      if isinstance(obj, np.floating):
          return float(obj)
      if isinstance(obj, np.ndarray):
          return obj.tolist()
      if isinstance(obj, set):
        return list(obj)
      return super(NpEncoder, self).default(obj)


def call_gemini_api(
    input_text, model="gemini-2.0-flash-exp", max_decode_steps=20, temperature=0.8
):
  assert isinstance(input_text, str)
  try:
    model = genai.GenerativeModel(model_name=model)

    result = model.generate_content(
      input_text,
      generation_config=genai.GenerationConfig(
          response_mime_type="application/json",
          response_schema=LinearRegressionSchema,
          temperature=temperature,
          max_output_tokens=max_decode_steps,
      ),
    ).candidates[0].content.parts[0].text
    print(result)
    return result

  except Exception as e:
    print(e)
    retry_time = 65  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_gemini_api(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )
  
def call_hf_model(
    input_text, tokenizer, model, max_decode_steps=20, temperature=0.8,
):
  assert isinstance(input_text, str)
  try:
    input_ids = tokenizer(input_text, return_tensors="pt").to(torch.device("mps"))

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])

  except Exception as e:
    print(e)
    retry_time = 65  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_hf_model(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )


def gen_meta_prompt(
    old_value_pairs_set,
    X,  # pylint: disable=invalid-name, unused-argument
    y,  # pylint: disable=unused-argument
    num_input_decimals=5,
    num_output_decimals=5,
    max_num_pairs=100,
):
  """Generate the meta-prompt for optimization.

  Args:
    old_value_pairs_set (set): the set of old (w, b, z) pairs.
    X (np.array): the 1D array of x values.
    y (np.array): the 1D array of y values.
    num_input_decimals (int): the number of decimals for (w, b) in the
      meta-prompt.
    num_output_decimals (int): the number of decimals for z in the meta-prompt.
    max_num_pairs (int): the maximum number of exemplars in the meta-prompt.

  Returns:
    meta_prompt (str): the generated meta-prompt.
  """
  old_value_pairs_set = set(
      [
          (
              np.round(w, num_input_decimals)
              if num_input_decimals > 0
              else int(w),
              np.round(b, num_input_decimals)
              if num_input_decimals > 0
              else int(b),
              np.round(z, num_output_decimals)
              if num_output_decimals > 0
              else int(z),
          )
          for w, b, z in old_value_pairs_set
      ]
  )
  old_value_pairs = list(old_value_pairs_set)
  old_value_pairs = sorted(old_value_pairs, key=lambda x: -x[2])[
      -max_num_pairs:
  ]

  old_value_pairs_substr = "".join(f"\ninput:\nw={w}, b={b}\nvalue:\n{z}\n" for w,b,z in old_value_pairs)
  meta_prompt = """
Now you will help me minimize a function with two input variables w, b. I have some (w, b) pairs and the function values at those points. The pairs are arranged in descending order based on their function values, where lower values are better.
  """.strip()
  meta_prompt += "\n\n"
  meta_prompt += old_value_pairs_substr.strip()
  meta_prompt += "\n\n"

  meta_prompt += """Give me a new (w_value, b_value) pair that is different from all pairs above, and has a function value lower than any of the above. Do not write code. The output should be in json format '{reasoning: reason, w_b_pair: tuple[int,int]}' where values are your predicted [w_value, b_value] pairs based on the reasoning provided as reason and w and b are values within the integer range (-100 to 100). Use this JSON schema: LinearRegressionSchema = {'reasoning': str, weight_bias_pair: List[int,int]} Return: LinearRegressionSchema""".strip()
  return meta_prompt
