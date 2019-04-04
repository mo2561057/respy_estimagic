# TODO

* cleanup, review
* parallel regression tests
* start age handling

## Challenges

A main challenge arises as the set of VALID MOMENTS might be different between the observed data and the simulated data. In addition, the set of VALID MOMENTS for the simulated data also potentially varies during the estimation. As an example, consider the descriptives of the wage by period. If for a given parameterization there is nobody working, then these are simply not defined.

---> We need to handle the case of an invalid evaluation of the criterion function

---> We need to decide whether we were able to evaluate the criterion function properly.

As this problem, and potentially its solution, is highly context dependent, I want to move all this into the get_moments() function if possible. This function should simply return a numpy array with the simulated moments. It then must serve two purposes as it also needs to determine whether a valid evaluation is possible in some way, thus it will depend on the observed moments. Maybe the easiest is to set all elements in the returned value to a HUGE_FLOAT thus resulting in a HUGE value of the criterion function and returning a boolean that indicates that an invalid evaluation is taking place and thus an extra log entry is made.

## Refactorings

I want to do only limited refactoring as most code is due to the poor design of the RESPY package and the change in the parameterization of the covariance matrix of the shocks. All this will be dealt with in NORPY.

There are also some tests that explicitly rely on the REPSY package thus there is no clear separation between the packages. Howevet, these tests are very useful so I will keep them for now and then deal with them when NORPY is integrated.

The ability to restart needs to be part of the adapter classes as it will be highly model dependent.
