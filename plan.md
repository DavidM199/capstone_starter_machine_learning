Columns:
- body_type
- diet
- drinks
- drugs
- education
- ethnicity
- height
- income
- job
- offspring
- orientation
- pets
- religion
- sex
- sign
- smokes
- speaks
- status



#### df.religion.value_counts():
agnosticism                                   2724
other                                         2691
agnosticism but not too serious about it      2636
agnosticism and laughing about it             2496
catholicism but not too serious about it      2318
atheism                                       2175
other and laughing about it                   2119
atheism and laughing about it                 2074
christianity                                  1957
christianity but not too serious about it     1952
other but not too serious about it            1554
judaism but not too serious about it          1517
atheism but not too serious about it          1318
catholicism                                   1064
christianity and somewhat serious about it     927
atheism and somewhat serious about it          848
other and somewhat serious about it            846
catholicism and laughing about it              726
judaism and laughing about it                  681
buddhism but not too serious about it          650
agnosticism and somewhat serious about it      642
judaism                                        612
christianity and very serious about it         578
atheism and very serious about it              570
catholicism and somewhat serious about it      548
other and very serious about it                533
buddhism and laughing about it                 466
buddhism                                       403
christianity and laughing about it             373
buddhism and somewhat serious about it         359
agnosticism and very serious about it          314
judaism and somewhat serious about it          266
hinduism but not too serious about it          227
hinduism                                       107
catholicism and very serious about it          102
buddhism and very serious about it              70
hinduism and somewhat serious about it          58
islam                                           48
hinduism and laughing about it                  44
islam but not too serious about it              40
judaism and very serious about it               22
islam and somewhat serious about it             22
islam and laughing about it                     16
hinduism and very serious about it              14
islam and very serious about it                 13


#### df.body_type.value_counts()
average           14652
fit               12711
athletic          11819
thin               4711
curvy              3924
a little extra     2629
skinny             1777
full figured       1009
overweight          444
jacked              421
used up             355
rather not say      198

#### df.income.value_counts()
-1          48442
 20000       2952
 100000      1621
 80000       1111
 30000       1048
 40000       1005
 50000        975
 60000        736
 70000        707
 150000       631
 1000000      521
 250000       149
 500000        48


#### df.status.value_counts()

single            55697
seeing someone     2064
available          1865
married             310
unknown              10





## Use Classification Techniques 

Can drug use and drinking predict seriousness of religion?
KNeighborsCalssifier, SVC



## Regression Techniques
Multiple Linear Regression
K Nearest Neighbors Regression


Can we predict income with average word length and drinking? 
No, we can not. Both models perform exceptionally bad, but now we can see that there isn't a strong correlation in tha data between staying sober and earning more. 
