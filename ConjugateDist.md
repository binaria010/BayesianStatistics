<head>
  <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
         displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
         processEscapes: true
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
  <meta name="google-site-verification" content="kuks5e4as6qBaGVCSzmHkQJa5Tss89_g5DmRXeUi7K8" />
</head>


<a href="index" class="previous"> &laquo; Previous</a>    <a href="HypothesisTest" class="next">Next &raquo;</a>  


<h1> Conjugate Distributions </h1>

We already saw in the <a href="index"> Introduction </a> that if one has a random sample $Y_1, \dots, Y_n$ such that 

$$
Y_i \vert \theta \sim Ber(\theta),\quad \text{and}\quad \theta \sim \mathcal{U}(0,1) = Beta(1,1)
$$

then the posterior distribution of $\theta$ is again a Beta distribution $Beta(1 + \sum_{i}y_i, 1 + n -\sum_{i}y_{i})$. We say that the Beta distribution os conjugate to the Binomial distribution.


In this section we will see this in more detail and a few more examples.

<h2> Prior Beta </h2>

Recall that if $X\sim Beta(\alpha, \beta)$ then its density function is
$$
f_X(x) = \frac{\Gamma(\alpha +\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta -1}{\bf 1}_{(0,1)}(x)
$$

Let $\theta \in \Theta$ a parameter of interest. Let $Y_1,\dots,Y_n$ be a $\theta$ random sample, that is, the variables are conditional on $\theta$ independent and identically distributed, with distribution given by $Y_i \vert\theta \sim Ber(\theta)$.

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

$$
\begin{align}
&\frac{\Gamma(\alpha_{prior} +\beta_{prior})}{\Gamma(\alpha_{prior})\Gamma(\beta_{prior})}\theta^{\alpha_{prior}-1}(1-\theta)^{\beta_{prior} -1} \theta^{\sum_{i}y_i}(1-\theta)^{n -\sum_{i}y_i} \\
  \\
=& \frac{\Gamma(\alpha_{prior} +\beta_{prior})}{\Gamma(\alpha_{prior})\Gamma(\beta_{prior})}\theta^{\alpha_{prior} + \sum_{i}y_i -1}(1-\theta)^{\beta_{prior}+ n -\sum_{i}y_i -1} 
\end{align}
$$

Meanwhile the denominator is simply the integral with respect to $\theta$ of the last expression above, this is because the posterior is a probability density. By use of the following property:

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



```python

```


```python

```
