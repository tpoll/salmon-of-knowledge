 java -cp $WEKA_INS/weka.jar weka.filters.unsupervised.attribute.Standardize \
   -b \
   -i yelp_maxent_training.arff \
   -o train_std.arff \
   -r yelp_maxent_test.arff \
   -s test_std.arff