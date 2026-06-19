#set page(numbering: "1")
#set heading(numbering: "1.a")
#set text(font: "New Computer Modern", size: 12pt)
#align(left, text(17pt)[
  *Problem Set #1: Supervised Learning*
])

= Linear Classifiers (logistic regression and GDA)

(a) Given the average empirical loss for logistic regression
$
  J(theta) = -1/m sum_(i=1)^m y^((i)) log(h_theta (x^((i)))) + (1 - y^((i)))log(1 - h_theta (x^((i))))
$
where $y^((i)) in {0, 1}, h_theta (x) = g(theta^T x)$ and $g(z) = frac(1, 1 + e^(-z))$

Then, the gradient of $J(theta)$ is
$
  gradient_theta J(theta)
  = 1/m sum_(i=1)^m [frac(y^((i)), h_theta (x^((i)))) - frac((1 - y^((i))), 1 - h_theta (x^((i))))] h_theta (x^((i))) (1 - h_theta (x^((i))) x^((i))
  = 1/m sum_(i=1)^m [y^((i)) - h_theta (x^((i)))] x^((i))
$

And the Hessian of $J(theta)$ is
$
  H
  = 1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] x^((i)) (x^((i)))^T
$

$forall z in RR^n, z^T H z
&= sum_(k=1)^n sum_(l=1)^n z_k H_(k l) z_l \
&= sum_(k=1)^n sum_(l=1)^n z_k [1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] x_k^((i)) x_l^((i))] z_l \
&= 1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] sum_(k=1)^n sum_(l=1)^n z_k x_k^((i)) x_l^((i)) z_l \
&= 1/m sum_(i=1)^m h_theta (x^((i))) [1 - h_theta (x^((i)))] (z^T x^((i)))^2 >= 0$

Thus $J(theta)$ is convex and hence if there is any critical point, it is its global minimum.



