## 存在的问题 
<details>
<summary>1. 观测数据的有的utc是有没有gt值的</summary>

这个好解决，直接删了没有真值的数据就行了

</details>

<details>
<summary>2. 每个历元观测到的卫星星座数量都不一样</summary>

多的50个，少的21个。这样对模型的输入就要求是不固定的。
就有点麻烦。目前的方法是用数据筛选的方法，筛成20颗。按照特定顺序排序。

</details>

<details>
<summary>3. 有的数据没有观测值</summary>

直接删了没有观测值的数据就行了，对齐一下gt和obs的颗粒度

</details>