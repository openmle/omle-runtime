#include <gtest/gtest.h>

#include <cmath>

#include "naive_bayes_node.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

// -----------------------------------------------------------------------
// Gaussian Naive Bayes
// -----------------------------------------------------------------------

TEST(NaiveBayes, Gaussian_PredictClass) {
  // Two classes, one feature.
  // Class 0: mean=0, var=1; class 1: mean=10, var=1
  // Equal priors: log_prior = log(0.5)
  auto node = make_naive_bayes_node(
      NaiveBayesVariant::Gaussian,
      {std::log(0.5f), std::log(0.5f)},  // class_log_priors
      {0.0f, 10.0f},                     // means
      {1.0f, 1.0f},                      // variances
      std::nullopt,                      // variance_epsilon
      {},                                // feature_log_prob
      std::nullopt,                      // binarize_threshold
      {},                                // category_log_prob
      {},                                // category_offset
      {},                                // category_count
      2,                                 // n_classes
      1);                                // n_features
  node->in_names = {"x"};
  node->out_names = {"pred", "prob"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 1);
  x.at(0, 0) = 0.0f;   // close to class 0
  x.at(1, 0) = 10.0f;  // close to class 1
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  EXPECT_FLOAT_EQ(vs.get("pred").at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(vs.get("pred").at(1, 0), 1.0f);

  // Prob[class] should sum to ~1
  const Tensor& prob = vs.get("prob");
  EXPECT_NEAR(prob.at(0, 0) + prob.at(0, 1), 1.0f, 1e-5f);
  EXPECT_NEAR(prob.at(1, 0) + prob.at(1, 1), 1.0f, 1e-5f);
  // Winning class should have probability > 0.5
  EXPECT_GT(prob.at(0, 0), 0.5f);
  EXPECT_GT(prob.at(1, 1), 0.5f);
}

TEST(NaiveBayes, Gaussian_PriorInfluence) {
  // Two classes, one feature, x=5 is equidistant from class means (0 vs 10).
  // Heavily bias prior toward class 1 → class 1 should win.
  auto node = make_naive_bayes_node(
      NaiveBayesVariant::Gaussian,
      {std::log(0.01f), std::log(0.99f)},  // class_log_priors
      {0.0f, 10.0f},                       // means
      {1.0f, 1.0f},                        // variances
      std::nullopt,                        // variance_epsilon
      {},                                  // feature_log_prob
      std::nullopt,                        // binarize_threshold
      {},                                  // category_log_prob
      {},                                  // category_offset
      {},                                  // category_count
      2,                                   // n_classes
      1);                                  // n_features
  node->in_names = {"x"};
  node->out_names = {"pred"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(1, 1);
  x.at(0, 0) = 5.0f;
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 1).ok());

  EXPECT_FLOAT_EQ(vs.get("pred").at(0, 0), 1.0f);
}

// -----------------------------------------------------------------------
// Multinomial Naive Bayes
// -----------------------------------------------------------------------

TEST(NaiveBayes, Multinomial_PredictClass) {
  // Two classes, two features.
  // class 0 strongly associated with feature 0; class 1 with feature 1.
  auto node = make_naive_bayes_node(
      NaiveBayesVariant::Multinomial,
      {std::log(0.5f), std::log(0.5f)},  // class_log_priors
      {},                                // means
      {},                                // variances
      std::nullopt,                      // variance_epsilon
      {
          // feature_log_prob [class, feature]
          std::log(0.9f),
          std::log(0.1f),  // class 0: mostly feature 0
          std::log(0.1f),
          std::log(0.9f),  // class 1: mostly feature 1
      },
      std::nullopt,  // binarize_threshold
      {},            // category_log_prob
      {},            // category_offset
      {},            // category_count
      2,             // n_classes
      2);            // n_features
  node->in_names = {"x"};
  node->out_names = {"pred"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 2);
  // sample 0: heavy on feature 0 → class 0
  x.at(0, 0) = 10.0f;
  x.at(0, 1) = 0.0f;
  // sample 1: heavy on feature 1 → class 1
  x.at(1, 0) = 0.0f;
  x.at(1, 1) = 10.0f;
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  EXPECT_FLOAT_EQ(vs.get("pred").at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(vs.get("pred").at(1, 0), 1.0f);
}

// -----------------------------------------------------------------------
// Bernoulli Naive Bayes
// -----------------------------------------------------------------------

TEST(NaiveBayes, Bernoulli_PredictClass) {
  auto node = make_naive_bayes_node(
      NaiveBayesVariant::Bernoulli,
      {std::log(0.5f), std::log(0.5f)},  // class_log_priors
      {},                                // means
      {},                                // variances
      std::nullopt,                      // variance_epsilon
      {
          // feature_log_prob
          std::log(0.8f),
          std::log(0.2f),
          std::log(0.2f),
          std::log(0.8f),
      },
      0.5,  // binarize_threshold
      {},   // category_log_prob
      {},   // category_offset
      {},   // category_count
      2,    // n_classes
      2);   // n_features
  node->in_names = {"x"};
  node->out_names = {"pred"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 2);
  x.at(0, 0) = 1.0f;
  x.at(0, 1) = 0.0f;  // feature 0 present → class 0
  x.at(1, 0) = 0.0f;
  x.at(1, 1) = 1.0f;  // feature 1 present → class 1
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  EXPECT_FLOAT_EQ(vs.get("pred").at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(vs.get("pred").at(1, 0), 1.0f);
}
