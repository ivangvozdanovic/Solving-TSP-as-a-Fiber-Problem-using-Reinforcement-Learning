# Solving-TSP-as-a-Fiber-Problem-using-Reinforcement-Learning


We can think of a fiber as a solution space to a discrete optimization problem. Instead of sampling, we can design algorithms that sweep through this solution space and look for optimal solutions satisfying certain conditions. Instead of gathering a sample, we can look for a optimal integer solution with lowest cost. More generally, the optimal solution should maximize or minimize a given objective function. On top of solving a goal oriented fiber sampling problem, we can also interpret and use the optimal policy $\pi^*$ by leveraging certain algebraic results. In short, in this chapter we want to show that solving a combinatorial optimization MDP, such as a Traveling Salesman Problem, will yield an optimal policy $\pi^*$ which encodes a subset of the Gr{\"o}bner basis. 

We consider all combinatorial optimization problems that can be specified using an underlying graph $G = (V,E,W)$ with a set of edges $E$, set of vertices $V$ and set of weights $W$. Each edge has an associated unique positive weight $w_e\in W$ and we assume we have access to all such weights. For such a setup, we define the vertex-edge incidence matrix $M$ as the constraint matrix. Furthermore, we define a cost vector $C$ to be a vector of edge weights $w_e\in E$ ordered lexicographically. Thus, 
\begin{align*}
    C = (w_{01},w_{02},\dots,w_{N-1N})^T,~\ |V| = N
\end{align*}
and so for any integer vector observation $x\in \mathcal{F}{M}(b)$, the cost of such observation is simply the sum of its weights, computed as $C(x) = C^Tx$.
