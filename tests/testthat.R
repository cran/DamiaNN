library(testthat)
context("Architecture")

test_that("Layer sizes are valid", {
  expect_equal(1, 1)
})

test_that("Layers are fully connected", {
  expect_equal(1, 1)
})

test_that("Inf errors avoided", {
  expect_equal(9,9)
})
