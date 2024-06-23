# Methods ODES resolve Documentation

## Método de Euler

El método de Euler es muy sencillo, se basa en la expansión de Taylor de la función $x(t)$. Tenemos
$$
\text{Expansión Taylor} \Rightarrow x(t+h) = x(t) + h\frac{dx}{dt} + \overbrace{ \frac{h^2}{2} \frac{d^2x}{dt^2} } ^{\epsilon} + O(h^3).
$$
Esto implica que para avanzar en el tiempo la función por un paso $h$, el cual suponemos que es lo suficientemente pequeño, basta con utilizar la ecuación
$$
\boxed{x(t + h) = x(t) + hf(x,t).}
$$
El error asociado con la aproximación **está ligado a la cantidad de veces que hagamos la aproximación**, es decir, al número de pasos en el tiempo que utilicemos en nuestra solución. Lo podemos estimar de la siguiente forma
$$
\sum\epsilon = \sum_{k=0}^{N-1}\frac{h^2}{2}\left. \frac{d^2x}{dt^2} \right|_{x_k, t_k} = \frac{h}{2}\sum_{k=0}^{N-1}h\left.\frac{df}{dt}\right|_{x_k, t_k}\\
\approx \frac{h}2\int_a^b\frac{df}{dt}d t = \frac{h}{2}\left[f_b - f_a\right].
$$
En la ecuación anterior asumimos que tomamos $N = (b-a)/h$ pasos temporales para llegar al punto final.

* El algoritmo toma la siguiente forma:
  - Empezar con $t = t_0$, $x = x_0$
  - Discretizar el tiempo en pasos temporales de forma equidistante con espaciamiento $h$, donde cada punto en el tiempo está denotado con $t_i$
  - Para cada punto en el tiempo encontrar $x$ utilizando el resultado de la iteración previa: $x_i = x_{i-1} + hf(x_i)$
## Método de Runge-Kutta 2$^{\rm do}$ Orden (RK2)

La idea del método RK2 es utilizar el punto medio para evaluar el método de Euler. Mientras que el método de Euler se aplica en el punto $t$ para evaluar la derivada para aproximar la función en el punto $x = t + h$, el método RK2 utiliza el punto medio $t + h/2$. 

De esta forma, se alcanza una mejor aproximación para el mismo valor de $h$.

El método se deriva aplicando la serie de Taylor alrededor del punto medio $t + h/2$ para obtener el valor de la función en el punto $x(t + h)$. Tenemos
$$
x(t + h) = x\left(t + \frac{h}{2}\right) + \frac{h}{2}\left(\frac{{\rm d}x}{{\rm d}t}\right)_{t+h/2} + \frac{h^2}{8}\left(\frac{{\rm d}^2x}{{\rm d}t^2}\right)_{t+h/2} + O(h^3).
$$
Similarmente, podemos hacer lo mismo para $x(t)$, tal que
$$
x(t) = x\left(t + \frac{h}{2}\right) - \frac{h}{2}\left(\frac{{\rm d}x}{{\rm d}t}\right)_{t+h/2} + \frac{h^2}{8}\left(\frac{{\rm d}^2x}{{\rm d}t^2}\right)_{t+h/2} + O(h^3).
$$
Al sustraer ambas ecuaciones obtenemos
$$
x(t + h) = x(t) + h\left(\frac{{\rm d}x}{{\rm d}t}\right)_{t+h/2} + O(h^3)
$$
Finalmente,
$$
\boxed{x(t + h) = x(t) + hf[x(t + h/2), t + h/2] + O(h^3)}.
$$
El término de orden $h^2$ desaparece y nuestra aproximación tiene un error de orden $h^3$. Recordemos que incrementar el orden del error por un orden de magnitud es muy beneficioso a nivel computacional. 

El único problema es que requerimos conocer el valor de la función en el punto medio $x(t + h/2)$, el cual desconocemos.

Para aproximar este valor utilizamos el método de Euler con un paso $h/2$, $(x + h/2) = x(t) + \frac{h}{2}f(x,t)$. De esta manera, obtenemos las ecuaciones del método RK2:
* $k_1 = hf(x,t),$
* $k_2 = hf\left(x + \frac{k_1}{2},t + \frac{h}{2}\right)$
* $x(t + h) = x(t) + k_2$

## Método de Runge-Kutta de 4$^{\rm to}$ Orden

La metodología anterior se puede aplicar aún a más puntos ubicados entre $x(t)$ y $x(t + h)$ realizando expansiones de Taylor. De esta forma se pueden agrupar términos de orden $h^3$, $h^4$, etc; para cancelar dichas expresiones. 

El problema de hacer esto es que las expresiones se vuelven más complicadas conforme incrementamos el orden de aproximación. En general, la regla de dedo es que el $4^{\rm to}$ orden corresponde al mejor compromiso entre complejidad y error de aproximación. Este método es el más utilizado comunmente para resolver ODEs. 

El álgebra para encontrar las ecuaciones de $4^{\rm to}$ orden es tediosa, pero el resultado final es
* $k_1 = hf(x, t)$,
* $k_2 = hf\left(x + \frac{k_1}{2}, t+\frac{h}2\right)$,
* $k_3 = hf\left(x + \frac{k_2}{2}, t+\frac{h}2\right)$,
* $k_4 = hf\left(x + k_3, t + h \right)$,
* $x(t+h) = x(t) + \frac{1}{6}(k_1 + 2 k_2 + 2k_3 + k_4)$.
 
