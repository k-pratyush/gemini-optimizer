# gemini-optimizer

Exploring the OPRO paper (https://arxiv.org/pdf/2309.03409) to train a linear regression model on Gemini 2.0 Flash with SGD baselines. 

### Setup
```
conda create -n gemini-opro python=3.10
conda activate gemini-opro
pip install -r requirements.txt
rename .env-example to .env and add your GEMINI API KEY
```

### Running the experiments
set w_true and b_true values inside `opro.py` to generate data points for experiments. Other experiment parameters are tunable in the `settings.py` file.
```
python opro_optimizer/opro.py
```

Running SGD baselines for the same w_true and b_true values.
```
python opro_optimizer/sgd.py
```

Generating results and metrics
```
python eval.py
```

### Observations and Results
 - The temperature, number of generated points per step, and num pairs params act as an exploration-exploitation control for the language model.
 - The number of pairs given in the meta-prompt correlate with model performance in extreme cases (eg: <2,30>, <36, -1> pairs). If the pair counts are too low, model settles and hovers over a local minima.
 - Exp 1 - Low number of steps for (15,14) can be attributed to gaussian random initialization of w,b pair between 10 and 20.
 - Exp 2 - Hypothesis - The temperature parameter alone does not promote exploration in learnable params in the language model.
 - Exp 4 - As shown by the results, model is not impacted since data is not fed into the language model, just the weight, bias, and loss value.
 - Structured outputs work better when adding a 'reasoning' key in the output. The model seems to steer towards optimal values when reasoning tokens are added to it's context.

#### Experiment 1 - Initial Baselines
Temperature:	1\
Num points:	50\
Max Steps:	500\
Num Reps:	5\
Num generated points per step:	8\
SGD Tolerance:	0.1\
SGD Learning Rate:	1.00E-06

<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" rowspan="3">w_true</th>
    <th class="tg-c3ow" rowspan="3">b_true</th>
    <th class="tg-c3ow" colspan="3">number of steps</th>
    <th class="tg-c3ow" colspan="2">number of unique (w,b) pairs explored</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="2">Gemini 2.0 Flash</th>
    <th class="tg-c3ow">Stochastic Gradient Descent</th>
    <th class="tg-c3ow" colspan="2">Gemini 2.0 Flash</th>
  </tr>
  <tr>
    <th class="tg-c3ow">mean</th>
    <th class="tg-c3ow">std</th>
    <th class="tg-c3ow">Count</th>
    <th class="tg-c3ow">mean</th>
    <th class="tg-c3ow">std</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">15</td>
    <td class="tg-c3ow">14</td>
    <td class="tg-c3ow">1.6</td>
    <td class="tg-c3ow">0.49</td>
    <td class="tg-c3ow">121</td>
    <td class="tg-c3ow">10.8</td>
    <td class="tg-c3ow">1.72</td>
  </tr>
  <tr>
    <td class="tg-c3ow">17</td>
    <td class="tg-c3ow">17</td>
    <td class="tg-c3ow">6.8</td>
    <td class="tg-c3ow">0.75</td>
    <td class="tg-c3ow">118903</td>
    <td class="tg-c3ow">23.4</td>
    <td class="tg-c3ow">4.22</td>
  </tr>
  <tr>
    <td class="tg-c3ow">16</td>
    <td class="tg-c3ow">10</td>
    <td class="tg-c3ow">5.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">104577</td>
    <td class="tg-c3ow">19.8</td>
    <td class="tg-c3ow">3.19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow">14</td>
    <td class="tg-c3ow">1.9</td>
    <td class="tg-c3ow">178500</td>
    <td class="tg-c3ow">37.4</td>
    <td class="tg-c3ow">9.7</td>
  </tr>
  <tr>
    <td class="tg-c3ow">25</td>
    <td class="tg-c3ow">23</td>
    <td class="tg-c3ow">10.67</td>
    <td class="tg-c3ow">4.5</td>
    <td class="tg-c3ow">195228</td>
    <td class="tg-c3ow">43.67</td>
    <td class="tg-c3ow">17.21</td>
  </tr>
  <tr>
    <td class="tg-c3ow">2</td>
    <td class="tg-c3ow">30</td>
    <td class="tg-c3ow">did not converge, hovers around w=4 and b=5 even after ~60 iterations</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">243943</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">36</td>
    <td class="tg-c3ow">-1</td>
    <td class="tg-c3ow">did not converge, hovers around w=34 and b=25 even after ~40 iterations</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">231261</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</tbody></table>


#### Experiment 2 - An attempt to converge extreme points
Temperature:	1.3\
Num Reps:	1

<table class="tg"><thead>
  <tr>
    <th class="tg-7btt" rowspan="3">w_true</th>
    <th class="tg-amwm" rowspan="3">b_true</th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">number of steps</span></th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">number of unique (w,b) pairs explored</span></th>
  </tr>
  <tr>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">Gemini 2.0 Flash</span></th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">Gemini 2.0 Flash</span></th>
  </tr>
  <tr>
    <th class="tg-bobw"><span style="font-weight:bold">mean</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">std</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">mean</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">std</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-jkyp">2</td>
    <td class="tg-2b7s">30</td>
    <td class="tg-8d8j" colspan="2">Does not converge</td>
    <td class="tg-8d8j">-</td>
    <td class="tg-8d8j">-</td>
  </tr>
  <tr>
    <td class="tg-jkyp">36</td>
    <td class="tg-2b7s">-1</td>
    <td class="tg-8d8j" colspan="2">Does not converge</td>
    <td class="tg-8d8j">-</td>
    <td class="tg-8d8j">-</td>
  </tr>
</tbody></table>


#### Experiment 3 - Finally converged
Temperature:	1.5\
Max num pairs:	35\
Num Reps:	1\
Num generated points per step:	15

<table class="tg"><thead>
  <tr>
    <th class="tg-7btt" rowspan="3">w_true</th>
    <th class="tg-amwm" rowspan="3">b_true</th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">number of steps</span></th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">number of unique (w,b) pairs explored</span></th>
  </tr>
  <tr>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">Gemini 2.0 Flash</span></th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">Gemini 2.0 Flash</span></th>
  </tr>
  <tr>
    <th class="tg-bobw"><span style="font-weight:bold">mean</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">std</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">mean</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">std</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-pb0m">2</td>
    <td class="tg-8d8j">30</td>
    <td class="tg-8d8j">24</td>
    <td class="tg-8d8j">0</td>
    <td class="tg-8d8j">168</td>
    <td class="tg-8d8j">0</td>
  </tr>
  <tr>
    <td class="tg-pb0m">36</td>
    <td class="tg-8d8j">-1</td>
    <td class="tg-8d8j">20</td>
    <td class="tg-8d8j">0</td>
    <td class="tg-8d8j">137</td>
    <td class="tg-8d8j">0</td>
  </tr>
</tbody></table>

#### Experiment 4 - Trying to converge with higher number of data points
Num Points	100\
Temperature	1.5\
Max num pairs	35\
Num Reps	1\
Num generated points per step	15

<table class="tg"><thead>
  <tr>
    <th class="tg-7btt" rowspan="3">w_true</th>
    <th class="tg-amwm" rowspan="3">b_true</th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">number of steps</span></th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">number of unique (w,b) pairs explored</span></th>
  </tr>
  <tr>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">Gemini 2.0 Flash</span></th>
    <th class="tg-bobw" colspan="2"><span style="font-weight:bold">Gemini 2.0 Flash</span></th>
  </tr>
  <tr>
    <th class="tg-bobw"><span style="font-weight:bold">mean</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">std</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">mean</span></th>
    <th class="tg-bobw"><span style="font-weight:bold">std</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-pb0m">2</td>
    <td class="tg-8d8j">30</td>
    <td class="tg-8d8j">19</td>
    <td class="tg-8d8j">0</td>
    <td class="tg-8d8j">107</td>
    <td class="tg-8d8j">0</td>
  </tr>
  <tr>
    <td class="tg-pb0m">36</td>
    <td class="tg-8d8j">-1</td>
    <td class="tg-8d8j">19</td>
    <td class="tg-8d8j">0</td>
    <td class="tg-8d8j">107</td>
    <td class="tg-8d8j">0</td>
  </tr>
</tbody></table>

------

#### TODO:
 - [x] Run baselines on the gemini models
 - [x] Try with structured json outputs
 - [x] Train with SGD model
 - [x] Train a linear regression model with varying number of data points
 - [ ] Rerun experiments 3 and 4 with higher repetitions
 - [ ] Train a neural network model on the same ‘linear’ data
 - [ ] Fit a sine curve using LLM optimizer
 - [ ] Fit data points with decimal values

------

References:
Google-Deepmind. (n.d.). GitHub - google-deepmind/opro: official code for “Large Language Models as Optimizers.” GitHub. https://github.com/google-deepmind/opro
