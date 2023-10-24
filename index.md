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



  
  <body>
    <h1>Bayesian Statistics: a Crash Course</h1>
<article>

<header>
  
  <h2>Table of contents</h2>
  <nav>
      <ol>
          <li><a href="#introbayesian"> Introduction to Bayesian Inference  </a></li>
          <li>Frequentist Approach </li>
              <ul> 
                <li> <a href = "HypothesisTest">  Hypothesis Test</a> </li>
              </ul>
          <li> Bayesian Approach</li>
              <ul>
                <li><a href="ConjugateDistributions">Conjugate Distributions</a></li>
              <li> Algorithms and Simulations</li>
                  <ol>
                    <li><a href = "Metropolis-Hastings"> Metropolis Hastings Algorithm </a></li>
                  </ol>
              </ul>
      </ol>
  </nav>
</header>

  <h2 id = "introbayesian">Introduction to Bayesian Inference  </h2>
      <p>
    Bayesian statistics is a tool in statistical inference in which a parameter to be estimated (or infered) is not consider 
    an unknown but deterministic quantity rather than a random variable itself. 
    <br> 
    The idea is that one has a prior believe of the parameter $\theta$ in question that is expressed in terms of what is called the 
    prior probability $p(\theta)$. This probability is updated to what is called as posterior probability via Bayes Theorem in the following way: 
    $$
      p(\theta|y) = \frac{p(y|\theta) p(\theta)}{\int_{\theta \in \Theta}p(y|\theta) p(\theta)\,d\theta}
    $$

    where the integral in the denominator is a normalizing constant in order for $p(\theta|y)$ to be a probability distribution.
  
  </p>



 <a href="HyporthesisTest" class="previous"> &laquo; Next</a>

  </body>