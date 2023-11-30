<html>

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


<a href="index" class="previous"> &laquo; Previous</a>    <a href="HypothesisTest" class="next">Next &raquo;</a>  


<h1> Conjugate Distributions </h1>

We already saw in the <a href="index">introduction </a> that if one has a random sample $Y_1, \dots, Y_n$ such that 

$$
\begin{equation*}
Y_i | \theta \sim Ber(\theta)\quad \text{and} \quad \theta \sim \mathcal{U}(0,1) = Beta(1,1)
\end{equation*}
$$



then the posterior distribution of $\theta$ is again a Beta distribution $Beta(1 + \sum_{i}y_i, 1 + n -\sum_{i}y_{i})$. We say that the Beta distribution is conjugate to the Binomial distribution.

In this section we will see this in more detail and a few more examples.

<span class = "definition"> Definition </span>  (Conjugate Distribution)<br>
A class $\mathcal{P}$ of prior distributions for $\theta$ is said to be conjugate to the sampling model $p(y_1,\dots, y_n \vert \theta)$ if

$$
p(\theta) \in \mathcal{P} \Rightarrow p(\theta \vert y_1, \dots, y_n) \in \mathcal{P}
$$



<h2> Binomial Model </h2>

For binomial sample data let's consider the following prior:

<h3> Prior Beta </h3>

Recall that if $X\sim Beta(\alpha, \beta)$ then its density function is

$$
f_X(x) = \frac{\Gamma(\alpha +\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta -1}{\bf 1}_{(0,1)}(x)
$$

Let $\theta \in \Theta$ a parameter of interest. Let $Y_1,\dots,Y_n$ be a $\theta$ random sample, that is, the variables are conditional on $\theta$ independent and identically distributed, with distribution given by $Y_i \vert \theta \sim Ber(\theta)$.

Suppose that our prior believe on $\theta$ is that $\theta \sim Beta(\alpha_{prior}, \beta_{prior})$ for some $\alpha_{prior}, \beta_{prior}>0$ to be specified, then the posterior distribution of $\theta$ computed using Bayes rule is:

$$
p(\theta \vert y_1, \dots, y_n) = \frac{p(y_1,\dots, y_n \vert \theta)p(\theta)}{p(y_1,\dots, y_n)}
$$

where $p(y_1,\dots, y_n):=\int_{0}^{1} p(y_1,\dots, y_n \vert \theta)p(\theta)\,d\theta$

Since $Y_1,\dots, Y_n$ is a $\theta$ random sample, then 
$$
p(y_1,\dots, y_n \vert \theta) = \prod_{i=1}^{n}p(y_i \vert \theta) = \prod_{i=1}^{n}\theta^{y_i}(1-\theta)^{1-y_i} = \theta^{\sum_{i}y_i}(1-\theta)^{n -\sum_{i}y_i}
$$
where by $\sum_{i}y_i$ we mean $\sum_{i=1}^n y_i$.

Therefore the numerator in the posterior is:
<a id = "expression">
$$
\begin{align}
&\frac{\Gamma(\alpha_{prior} +\beta_{prior})}{\Gamma(\alpha_{prior})\Gamma(\beta_{prior})}\theta^{\alpha_{prior}-1}(1-\theta)^{\beta_{prior} -1} \theta^{\sum_{i}y_i}(1-\theta)^{n -\sum_{i}y_i} \notag\\
  \notag\\
=& \frac{\Gamma(\alpha_{prior} +\beta_{prior})}{\Gamma(\alpha_{prior})\Gamma(\beta_{prior})}\theta^{\alpha_{prior} + \sum_{i}y_i -1}(1-\theta)^{\beta_{prior}+ n -\sum_{i}y_i -1} \tag{1}
\end{align}
$$
</a>




Meanwhile the denominator is simply the integral with respect to $\theta$ of the last expression above [(1)](#expression) , this is because the posterior is a probability density. By use of the following property:

$$
\int_{0}^1 x^{a-1}(1-x)^{b-1} \,dx= \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)} 
$$


we conclude that the denominator is:

$$
p(y_1,\dots, y_n) = \frac{\Gamma(\alpha_{prior}+\sum_{i}y_i)\Gamma(\beta_{prior} +n -\sum_{i}y_i)}{\Gamma(\alpha_{prior} + \beta_{prior} +n)}.
$$

Thus the final expression for the posterior distribution is:

$$
p(\theta \vert y_1,\dots y_n) = \frac{\Gamma(\alpha_{prior} + \beta_{prior} +n)}{\Gamma(\alpha_{prior}+\sum_{i}y_i)\Gamma(\beta_{prior} +n -\sum_{i}y_i)}\theta^{\alpha_{prior} + \sum_{i}y_i -1}(1-\theta)^{\beta_{prior}+ n -\sum_{i}y_i -1} {\bf 1}_{[0,1]}(\theta).
$$

In other words 
$$
\theta\sim Beta(\alpha_{post}, \beta_{post}) \quad \text{with} \quad \alpha_{post} =\alpha_{prior} + \sum_{i}y_i, \quad \text{and} \quad \beta_{post}= \beta_{prior} + n - \sum_{i}y_i
$$


For this posterior distribution we will have the posterior expected mean and posterior mode to be:

$$
\begin{align*}
\mathbb{E}(\theta \vert y_1, \dots, y_n) = & \frac{\alpha_{post}}{\alpha_{post} +\beta_{post}} \\
= & \frac{\alpha_{prior} + \sum_{i}y_i}{\alpha_{prior} + \beta_{prior} + n }\\
= & \frac{n}{\alpha_{prior} + \beta_{prior} + n}\bar{y}_n + \frac{\alpha_{prior} + \beta_{prior}}{\alpha_{prior} + \beta_{prior} + n}\frac{\alpha_{prior}}{\alpha_{prior} +\beta_{prior}}\\
\end{align*}
$$

where $\bar{y}_n = \frac{\sum_{i=1}^n y_i}{n}$.

That is, the posterior expected value is a weighted average between the sample mean $\bar{y}_n$ and the prior expected value. Due to this weighted average, the quantity $\alpha_{prior}+\beta_{prior}$ is called sometimes the *effective size* while of course, $n$ is the sample size.

<h3> Prediction </h3>

An important feature in this approach is the existence of a  predictive distribution. Let us assume we are in the same case as before, where the prior for $\theta$ is $Beta(1,1)$ and the data is, conditionally on $\theta$ iid with $Y_i \vert \theta \sim Ber(\theta)$ for $i=1, \dots, n$. 

Now, consider a new datapoint, one that has not be seen before, let's call this new observation $Y_{new}$ (we use capital letters since we are assuming this has not been observed so it is random) from the same population, then the *predictive distribution* of $Y_{new}$ can be computed as follows:

$$
\begin{align*}
p_{Y_{new}}(y |y_1,\dots, y_n) = & \int_{\theta \in \Theta}p(y, \theta \vert y_1,\dots, y_n)\,d\theta \\
= & \int_{\theta \in \Theta}p(y \vert \theta, y_1,\dots, y_n)p(\theta \vert y_1,\dots, y_n)\,d\theta \\
= & \int_{\theta \in \Theta} \theta p(\theta \vert y_1,\dots, y_n)\,d\theta = \mathbb{E}(\theta \vert y_1, \dots, y_n)
\end{align*}
$$

In this case, we will have:

$$
\mathbb{P}(Y_{new} = 1\vert y_1,\dots, y_n) = \frac{n}{2 + n}\bar{y}_n + \frac{2}{2 + n}\frac{1}{2}
$$


<a href="WorkedExamples"> <h3> Worked Examples </h3> </a> 

</html>
