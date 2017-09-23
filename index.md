# StackExchange Tagger

The goal of our project is to develop an accurate tagger for questions posted on Stack Exchange. Our problem is an instance of the more general problem of developing accurate classifiers for large scale text datasets. We are tackling the multilabel classification problem where each item (in this case, question) can belong to multiple classes (in this case, tags). We are predicting the tags (or keywords) for a particular Stack Exchange post given only the question text and the title of the post. In the process, we compare the performance of Support Vector Classification (SVC) for different kernel functions, loss function, etc. 

We found linear SVC with Crammer Singer technique produces best results. 

# Some Results

Testing Error for SVC with different kernel functions where number of iterations = 10,000

|  Kernel          | C = 1000(hard-margin) | C = 0.001(soft-margin)   |
|------------------|-----------------------|--------------------------|
| RBF              |  43.1 %               |   48.5 %                 |
| Linear           |  51.9 %               |   45.2 %                 |
| Polynomial (n=2) |  54.4 %               |   65 %                   |
| Polynomial (n=3) |  72.2 %               |   84.4 %                 |
| Sigmoid          |  84.4 %               |   84.4 %                 |


Testing Error for Linear SVC with different techniques where C = 0.001 (soft-margin) and number of iterations = 10,000

| Technique      | Hinge Loss Function | Square Hinge Loss Function |
|----------------|---------------------|----------------------------|
| One-vs-rest    |      47.59 %        |            68 %            |
| Crammer Singer |      45.25 %        |           45.25%           |


#Report

Our detailed report and results are available [here](https://sites.google.com/site/sanketmehtaiitr/home/stack-exchange-tagger).


#Team

* [Sanket Mehta](https://twitter.com/sanketvmehta)
* [Shagun Sodhani](https://twitter.com/shagunsodhani)

This work has been done as a part of a course project for Artificial Neural Network (IEE-03) at IIT Roorkee. 
