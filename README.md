# PAC-Bayes bound for weighting ensembles

The following code base provides way to optimize ensemble weighting for predictors, as well as generalization bound to calculate garantees for the performance of any such weighting (see )

The code is structured in the following way;
`/models` contains ways of producing `.npz` files that can evaluated by `bounds.py`.
The implementation of `bounds.py` relies on previous work (by ??) contained in `/mvb`.

