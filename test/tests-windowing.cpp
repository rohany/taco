#include "test.h"
#include "taco/tensor.h"
#include "taco/codegen/module.h"
#include "taco/index_notation/index_notation.h"
#include "taco/lower/lower.h"

using namespace taco;

// Basic test of tensor operations on windows of tensors in different formats.
TEST(windowing, basic) {
    Tensor<int> expectedAdd("expectedAdd", {2, 2}, {Dense, Dense});
    expectedAdd.insert({0, 0}, 14);
    expectedAdd.insert({0, 1}, 17);
    expectedAdd.insert({1, 0}, 17);
    expectedAdd.insert({1, 1}, 20);
    expectedAdd.pack();
    Tensor<int> expectedMul("expectedMul", {2, 2}, {Dense, Dense});
    expectedMul.insert({0, 0}, 64);
    expectedMul.insert({0, 1}, 135);
    expectedMul.insert({1, 0}, 135);
    expectedMul.insert({1, 1}, 240);
    expectedMul.pack();
    Tensor<int> d("d", {2, 2}, {Dense, Dense});

    // These dimensions are chosen so that one is above the constant in `mode_format_dense.cpp:54`
    // where the known stride is generated vs using the dimension.
    // TODO (rohany): Change that constant to be in a header file and import it here.
    for (auto& dim : {6, 20}) {
        for (auto &x : {Dense, Sparse}) {
            for (auto &y : {Dense, Sparse}) {
                for (auto &z : {Dense, Sparse}) {
                    Tensor<int> a("a", {dim, dim}, {Dense, x});
                    Tensor<int> b("b", {dim, dim}, {Dense, y});
                    Tensor<int> c("c", {dim, dim}, {Dense, z});
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < dim; j++) {
                            a.insert({i, j}, i + j);
                            b.insert({i, j}, i + j);
                            c.insert({i, j}, i + j);
                        }
                    }
                    a.pack();
                    b.pack();
                    c.pack();

                    a.sliceMode(1, 2, 4); a.sliceMode(2, 2, 4);
                    b.sliceMode(1, 4, 6); b.sliceMode(2, 4, 6);
                    c.sliceMode(1, 1, 3); c.sliceMode(2, 1, 3);

                    IndexVar i, j;
                    d(i, j) = a(i, j) + b(i, j) + c(i, j);
                    d.evaluate();
                    ASSERT_TRUE(equals(expectedAdd, d))
                                                << endl << expectedAdd << endl << endl << d << endl
                                                << dim << " " << x << " " << y << " " << z << endl;

                    d(i, j) = a(i, j) * b(i, j) * c(i, j);
                    d.evaluate();
                    ASSERT_TRUE(equals(expectedMul, d))
                                                << endl << expectedMul << endl << endl << d << endl
                                                << dim << " " << x << " " << y << " " << z << endl;
                }
            }
        }
    }
}

// Test that operations can write to a window within an output tensor.
TEST(windowing, slicedOutput) {
    auto dim = 10;
    Tensor<int> expected("expected", {10, 10}, {Dense, Dense});
    expected.insert({8, 8}, 12);
    expected.insert({8, 9}, 14);
    expected.insert({9, 8}, 14);
    expected.insert({9, 9}, 16);
    expected.pack();
    for (auto& x : {Dense, Sparse}) {
        for (auto& y : {Dense, Sparse}) {
            Tensor<int> a("a", {dim, dim}, {Dense, x});
            Tensor<int> b("b", {dim, dim}, {Dense, y});
            Tensor<int> c("c", {dim, dim}, {Dense, Dense});
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    a.insert({i, j}, i + j);
                    b.insert({i, j}, i + j);
                }
            }
            a.pack();
            b.pack();

            // Slice a, b, and c.
            a.sliceMode(1, 2, 4); a.sliceMode(2, 2, 4);
            b.sliceMode(1, 4, 6); b.sliceMode(2, 4, 6);
            c.sliceMode(1, 8, 10); c.sliceMode(2, 8, 10);

            IndexVar i, j;
            c(i, j) = a(i, j) + b(i, j);
            c.evaluate();
            ASSERT_TRUE(equals(expected, c))
                                        << endl << expected << endl << endl << c << endl
                                        << dim << " " << x << " " << y << endl;
        }
    }
}

// Test how windowing interacts with sparse iteration space transformations and
// different mode formats.
// TODO (rohany): This test currently doesn't test many different transformations.
TEST(windowing, transformations) {
  auto dim = 10;
  Tensor<int> expected("expected", {2, 2}, {Dense, Dense});
  expected.insert({0, 0}, 12);
  expected.insert({0, 1}, 14);
  expected.insert({1, 0}, 14);
  expected.insert({1, 1}, 16);
  expected.pack();

  IndexVar i("i"), j("j"), i1 ("i1"), i2 ("i2");
  auto testFn = [&](std::function<IndexStmt(IndexStmt)> modifier, std::vector<Format> formats) {
    for (auto& format : formats) {
      Tensor<int> a("a", {dim, dim}, format);
      Tensor<int> b("b", {dim, dim}, format);
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          a.insert({i, j}, i + j);
          b.insert({i, j}, i + j);
        }
      }
      a.pack(); b.pack();
      a.sliceMode(1, 2, 4); a.sliceMode(2, 2, 4);
      b.sliceMode(1, 4, 6); b.sliceMode(2, 4, 6);

      Tensor<int> c("c", {2, 2}, {Dense, Dense});
      c(i, j) = a(i, j) + b(i, j);
      auto stmt = c.getAssignment().concretize();
      c.compile(modifier(stmt));
      c.evaluate();
      ASSERT_TRUE(equals(c, expected)) << endl << c << endl << expected << endl << format << endl;
    }
  };

  std::vector<Format> allFormats = {{Dense, Dense}, {Dense, Sparse}, {Sparse, Dense}, {Sparse, Sparse}};
  testFn([&](IndexStmt stmt) {
    return stmt.split(i, i1, i2, 4).unroll(i2, 4);
  }, allFormats);

  // TODO (rohany): Can we only reorder these loops in the Dense,Dense case? It seems so.
  testFn([&](IndexStmt stmt) {
    return stmt.reorder(i, j);
   }, {{Dense, Dense}});

  // We can only (currently) parallelize the outer dimension loop if it is dense.
  testFn([&](IndexStmt stmt) {
    return stmt.parallelize(i, taco::ParallelUnit::CPUThread, taco::OutputRaceStrategy::NoRaces);
  }, {{Dense, Dense}, {Dense, Sparse}});
}

TEST(windowing, immutableSlicing) {
  Tensor<int> expected("expected", {2, 2}, {Dense, Dense});
  expected.insert({0, 0}, 12);
  expected.insert({0, 1}, 14);
  expected.insert({1, 0}, 14);
  expected.insert({1, 1}, 16);
  expected.pack();

  auto dim = 6;
  Tensor<int> a("a", {dim, dim}, {Dense, Dense});
  Tensor<int> b("b", {dim, dim}, {Dense, Dense});
  Tensor<int> c("c", {2, 2}, {Dense, Dense});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a.insert({i, j}, i + j);
      b.insert({i, j}, i + j);
    }
  }
  a.pack(); b.pack();

  // TODO (rohany): We can add a slice object and accept a vector of slices
  //  so that many chained calls don't need to be made.
  auto aw = a.sliceModeImmut(1, 2, 4).sliceModeImmut(2, 2, 4);
  auto bw = b.sliceModeImmut(1, 4, 6).sliceModeImmut(2, 4, 6);

  IndexVar i, j;
  c(i, j) = aw(i, j) + bw(i, j);
  c.evaluate();
  cout << c.getSource() << endl;
  ASSERT_TRUE(equals(expected, c))
                << endl << expected << endl << endl << c << endl;
}
