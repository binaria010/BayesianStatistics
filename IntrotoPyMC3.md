<head>
  <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { tags: 'ams' } } }); </script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        extensions: ["amsthm.js", "AMSmath.js","AMSsymbols.js", "autobold.js"],
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
         displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
         processEscapes: true
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript">
  </script>
  <meta name="google-site-verification" content="kuks5e4as6qBaGVCSzmHkQJa5Tss89_g5DmRXeUi7K8" />

  <style> 
  body{
    .previous {
    background-color: #f1f1f1;
    color: black;
    }
    .next {
      background-color: #04AA6D;
      color: white;
    }
    .example,.theorem,.lemma,.problem, .definition {
       font-weight:bold; 
    }
  }
  </style>
</head>

<a href="MetropolisHastingsAlg" class="previous"> &laquo; Back</a>  <a href="index" class="next"> &raquo; Table of Contents</a> 

<h1>A Quick Intro to PyMC3 and ArviZ libraries</h1>


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3
import arviz
import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
```

The following data consists of the weight in pounds of 48 specimens of penguins from Antartica. The goal is to make some bayesian inference on the mean and variance of the weight.


```python
file_name = 'data/weights.csv'

data = np.loadtxt(file_name)
data[0:5]
```




    array([51.06, 55.12, 53.73, 50.24, 52.05])




```python
len(data)
```




    48




```python
arviz.plot_kde(data, rug=True,)
plt.show()

```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_7_0.png)
    


The KDE plot of this dataset suggests that,  except for a couple of observations, a bell shape distribution is a good approximation for the likelihood of this data. 

In this example we are going to make some Bayesian inference on the mean and variance of this distribution. First we are going to ignore the two 'outliers' and consider a Normal distribution for the likelihood. Then we will consider ommiting these two observations and compare both inference analysis.



Since we have no prior idea on the mean $\mu$ and the variance $\sigma^2$ of the data we will set non informative priors on both parameters. For the mean we will take as prior a uniform distirbution in the range of the data itself and for the variance we will consider the absolute value of a normal distribution, this in order to obtain a random variable with support $\mathbb{R}_{\geq 0}$. This distribution is known as *Half Normal* and it is already implemented as part of the distributions in **PyMC3**.

The model then will be of the type:

$$
\begin{align*}
\mu \sim \mathcal{U}(a, b)\\
\sigma \sim \vert \mathcal{N}(0, \sigma_0^2)\vert \\
y \sim \mathcal{N}(\mu, \sigma^2)
\end{align*}
$$

with $a = \min(data), b = \max(data)$ and $\sigma_0 = 50$, for this latter parameter we can set any ad-hoc value since we have no prior knowledge on the variance. 

Now, let us set the model, which we will call $model_0$, in **PyMC3**:


```python
a = np.floor(np.min(data) - 10)
b = np.ceil(np.max(data) + 10)
sigma_0 = 100

with pymc3.Model() as model_0:
    mu_prior = pymc3.Uniform('$\mu$',lower = a, upper= b)
    sigma_prior = pymc3.HalfNormal('$\sigma$', sd = sigma_0)
    y_likelihood = pymc3.Normal('y', mu = mu_prior, sd = sigma_prior, observed = data)
    infered_data = pymc3.sample(draws=1000, chains = 3, return_inferencedata=True)
```


```python
infered_data
```

Now, let us plot the results:


```python
arviz.plot_trace(infered_data, compact=False, figsize=(12, 6))
plt.show()
```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_14_0.png)
    


PymC3 runs three Markov chains for us and we can see in the plot above the KDE's of each chain for each parameter (left) and the trajectories or values of each draw.  By setting the parameter *combine* to True, the plot_trace function combines the three chains into one. Let's see the result of doing this:


```python
arviz.plot_trace(infered_data, combined = True, compact=False, figsize=(12, 6))
plt.show()
```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_16_0.png)
    


The function *summary* from ArviZ library gives us a summary of the results of the MCMC algorithm. It returns a pandas data frame with all the statistics of the Markov Chain such as the estimated mean and standard deviation for each parameter, the 94% high density interval although we can ask for other sizes by setting the *hdi_prob* to the required value, the Markov chain standard error for the mean and standar deviation estimations, the effective sample size computed with two methods (ess_bulk, ess_tail) which tell us the amount of independent samples would be necessary to obtained similar estimations; the nearer this number is to the total of samples the better. Finally the r_hat which tell us the ratio of variability in the chains to the variability within the chains.

The plot above for the KDE for each parameter correspond to the estimation of the marginal densities. We can also plot the KDE of the joint density with the function *plot_pair* form Arviz:


```python
arviz.plot_pair(infered_data, kind = 'kde',  marginals = True, )
plt.show()
```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_19_0.png)
    


If we wanted to get access to the drwas of each parameter we can index the posterior and indicate the variable (in this case we have two) and also the chain from which we want the samples.


```python
infered_data.posterior['$\mu$'][0].to_numpy()[0:10]
```




    array([53.35476037, 53.45003486, 53.37961168, 53.62442727, 53.64037276,
           53.49560377, 53.49560377, 53.79726556, 53.42916389, 53.41942054])



We can combine the samples from the three chains into a big numpy array:


```python
mu_samples = infered_data.posterior['$\mu$'][0:3].to_numpy()
```


```python
mu_samples.shape
```




    (3, 1000)



<h3> Posterior Predictive Checks </h3>

Now, with the posterior samples obtained via the MCMC we can simulate new data and check how consistent this simulated data is with respect to the observed data. This is a kind of diagnosis of the model since we are going to use these posterior samples to make some predictions and use them to check the model.

Recall that the posterior predictive distribution is

$$
p(y_{new}\vert y) =\int_{\theta\in\Theta} p(y_{new}\vert \theta)p(\theta\vert y)\,d\theta
$$
that is, it is an average of conditional predictions over the $\theta$ parameter.

With the function *sample_posterior_predictive* we generate of sets samples of the variable $Y_{new}$ each of these set with the size of the original data:


```python
y_pred_model0 = pymc3.sample_posterior_predictive(trace = infered_data, model=model_0, keep_size=True)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='3000' class='' max='3000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [3000/3000 00:01&lt;00:00]
</div>



This function returns a Dictionary object with keys the name of the observed variable which we  called 'y' and the value is a numpy array of shape $(chains, draws, len(data))$, this because we set the parameter *keep_size* equals to True in order to be able to make a posterior predictive check (ppc) plot 


```python
print(y_pred_model0['y'].shape)
```

    (3, 1000, 48)



```python
y_pred_model0['y'][0,0,0:10]
```




    array([58.14823624, 57.68249386, 46.18514987, 55.28961718, 57.00301707,
           57.16203663, 54.3395682 , 57.83787608, 53.90723813, 53.9189246 ])



Now, we add these posterior predictive samples to the object *infered_data* using the function *concat* from Arviz:


```python
ppc_data = arviz.concat(infered_data, arviz.from_dict(posterior_predictive= y_pred_model0), inplace=False)
ppc_data
```

Check that in fact these posterior predictive samples are the ones saved in *y_pred_model0*:


```python
ppc_data.posterior_predictive['y'].to_numpy()[0,0, 0:10]
```




    array([58.14823624, 57.68249386, 46.18514987, 55.28961718, 57.00301707,
           57.16203663, 54.3395682 , 57.83787608, 53.90723813, 53.9189246 ])



Now, let us make the *ppc* plot:


```python

arviz.plot_ppc(data=ppc_data,num_pp_samples= 30, data_pairs={"y":"y"}, color='purple', random_seed= 33)
plt.show()

```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_35_0.png)
    


The curves in light purple are the KDE plots of num_pp_samples =10 sets of 48 samples randomly chosen from the posterior predictive simulated data. The orange dashed curve is the average of all the light purple curves. The black curve is the KDE plot of the observed data. What one hopes to see in this plot is that the light curves surrounds the black curve. This provides confidence that our model is able to produce data similar to the observed one.

Let us make tha same analysis but dropping out the two points from the observed data that seemed to be outliers. This of course is completely adhoc since we don't really know whether they are outliers or not.


```python
mask = data > 60
sum(mask)
```




    2




```python
data_reduced = data[~mask]
print(len(data), len(data_reduced))

arviz.plot_kde(data_reduced, figsize=(12,4), rug = True)
plt.show()
```

    48 46



    
![png](IntrotoPyMC3_files/IntrotoPyMC3_38_1.png)
    



```python
with pymc3.Model() as model_1:
    mu_prior = pymc3.Uniform('$\mu$',lower = a, upper= b)
    sigma_prior = pymc3.HalfNormal('$\sigma$', sd = sigma_0)
    y_likelihood = pymc3.Normal('y', mu = mu_prior, sd = sigma_prior, observed = data_reduced)
    infered_data_1 = pymc3.sample(draws=1000, chains = 3, return_inferencedata=True)

y_pred_model1 = pymc3.sample_posterior_predictive(trace = infered_data_1, model=model_1, keep_size=True)

ppc_data_reduced = arviz.concat(infered_data_1, arviz.from_dict(posterior_predictive= y_pred_model1), inplace=False)
ppc_data_reduced

```


```python

arviz.plot_pair(infered_data_1, kind = 'kde',  marginals = True, )
plt.show()


```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_40_0.png)
    



```python


_, ax = plt.subplots(1, 2, sharex=True, sharey=True , figsize=(12, 4))
arviz.plot_ppc(data=ppc_data,num_pp_samples= 50, data_pairs={"y":"y"}, color='purple', random_seed= 33, ax = ax[0])
arviz.plot_ppc(data=ppc_data_reduced,num_pp_samples= 50, data_pairs={"y":"y"}, color='purple', random_seed= 33, ax=ax[1])
ax[0].set_title("PPC for Complete data")
ax[1].set_title("PPC for data without outliers")
ax[1].legend().set_visible(False)
plt.show()
```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_41_0.png)
    


<h2> Changing the likelihood </h2>

The plot comparisson above shows us that by dropping out the "outliers" doesn't seem to improve the model. We could try another approach. We can keep those two data points and consider as a model for the data a distribution with heavier tails, for instance a T students distribution. The problem with the normal model is that due to the presence of these two outliers, it tends to move to the right.

Let us now consider a new model with likelihood given by a T-distribution:

$$
\begin{align*}
\mu &\sim \mathcal{U}(a, b)\\
\sigma &\sim \vert \mathcal{N}(0, \sigma_0^2)\vert \\
\nu &\sim \mathcal{E}(\lambda)\\
y &\sim \mathcal{T}(\mu, \sigma, \nu)
\end{align*}
$$

Because the T distribution has 3 parameters: the mean, the scale and the degrees of freedom $\nu$, we need to especify also a prior for $\nu$.

As before, take $a = \min(data), b = \max(data)$ and $\sigma_0 = 50$, for this latter parameter we can set any ad-hoc value since we have no prior knowledge on the variance and since a T distribution with big degree of freedom is pretty similar to a Gaussia then consider the mean of the prior for $\nu$ to be a relatively big number, let's say $1/\lambda = 20$.




```python
a = np.min(data)
b = np.max(data)
sigma_0 = 10
lambd = 1/30

with pymc3.Model() as model_T:
    mu = pymc3.Uniform('$\mu$', lower = a, upper = b)
    sigma = pymc3.HalfNormal('$\sigma$', sd = sigma_0)
    nu = pymc3.Exponential('$\\nu$', lam = lambd)
    y = pymc3.StudentT('y', mu = mu, sd = sigma, nu= nu, observed = data)
    trace = pymc3.sample(draws = 1000, chains = 2, return_inferencedata=True)

```


```python
arviz.plot_trace(data = trace, compact=False, figsize = (12, 10))
plt.show()

```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_44_0.png)
    


Let's compare the summary for both models:


```python
arviz.summary(trace)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$\mu$</th>
      <td>53.033</td>
      <td>0.388</td>
      <td>52.301</td>
      <td>53.743</td>
      <td>0.011</td>
      <td>0.008</td>
      <td>1185.0</td>
      <td>1156.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>$\sigma$</th>
      <td>2.184</td>
      <td>0.419</td>
      <td>1.390</td>
      <td>2.942</td>
      <td>0.017</td>
      <td>0.012</td>
      <td>624.0</td>
      <td>859.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>$\nu$</th>
      <td>4.764</td>
      <td>5.111</td>
      <td>0.933</td>
      <td>9.926</td>
      <td>0.204</td>
      <td>0.145</td>
      <td>826.0</td>
      <td>752.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
arviz.summary(infered_data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$\mu$</th>
      <td>53.498</td>
      <td>0.504</td>
      <td>52.494</td>
      <td>54.404</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>2439.0</td>
      <td>1963.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>$\sigma$</th>
      <td>3.540</td>
      <td>0.379</td>
      <td>2.825</td>
      <td>4.197</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>2077.0</td>
      <td>2037.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's make the posterior predictive check of the T-model:


```python
y_ppc_T = pymc3.sample_posterior_predictive(trace = trace, model=model_T, random_seed=123, keep_size=True)
data_ppc_T = arviz.concat(trace, arviz.from_dict(posterior_predictive=y_ppc_T), inplace=False)

```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:01&lt;00:00]
</div>




```python
arviz.plot_ppc(data_ppc_T, num_pp_samples= 50, random_seed=40)
plt.xlim((40, 70))
plt.show()
```


    
![png](IntrotoPyMC3_files/IntrotoPyMC3_50_0.png)
    


In terms of the location of the mean the T model seems to be better than the Gaussian, also in terms of the spread.

<br>
<br>
<br>

<a href="MetropolisHastingsAlg" class="previous"> &laquo; Back</a>  <a href="index" class="next"> &raquo; Table of Contents</a> 
