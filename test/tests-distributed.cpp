#include "test.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/distribution.h"
#include "taco/lower/lower.h"
#include "codegen/codegen.h"
#include "codegen/codegen_legion_c.h"
#include "codegen/codegen_legion_cuda.h"
#include "codegen/codegen_cuda.h"

#include "taco/index_notation/transformations.h"
#include "taco/index_notation/provenance_graph.h"

#include <fstream>

using namespace taco;

const int NNZ_PER_THREAD=8;
const int WARP_SIZE = 32;
const int BLOCK_SIZE=256;
const int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
const int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;

TEST(distributed, test) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Dense});
  Tensor<int> c("c", {dim}, Format{Dense});

  IndexVar i("i"), in("in"), il("il"), il1 ("il1"), il2("il2");
  a(i) = b(i) + c(i);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i}, {in}, {il}, Grid(4));

  // Communication modification must go at the end.
  // TODO (rohany): name -- placement
//  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), il1).pushCommUnder(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), in);
  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), in).pushCommUnder(c(i), in);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, cuda_test) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Dense});
  IndexVar i("i"), in("in"), il("il"), il1 ("il1"), il2("il2");
  a(i) = b(i);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i}, {in}, {il}, Grid(4));
  int NNZ_PER_THREAD=8;
  int WARP_SIZE = 32;
  int BLOCK_SIZE=256;
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
  IndexVar f1("f1"), f2("f2"), f3("f3"), f4("f4"), block("bvar"), warp("wvar"), thread("tvar");
  stmt = stmt.split(il, block, f1, NNZ_PER_TB)
      .split(f1, warp, f2, NNZ_PER_WARP)
      .split(f2, thread, f3, NNZ_PER_THREAD)
      .parallelize(block, ParallelUnit::GPUBlock, taco::OutputRaceStrategy::IgnoreRaces)
      .parallelize(warp, ParallelUnit::GPUWarp, taco::OutputRaceStrategy::IgnoreRaces)
      .parallelize(thread, ParallelUnit::GPUThread, taco::OutputRaceStrategy::IgnoreRaces)
      ;
  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), in);
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
  {
    ofstream f("../legion/cuda-test/taco-generated.cu");
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
    f.close();
  }
}

TEST(distributed, multiDim) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i, j}, {in, jn}, {il, jl}, Grid(4, 4));
  stmt = stmt.pushCommUnder(a(i, j), jn).pushCommUnder(b(i, j), jn);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, basicComputeOnto) {
  int dim = 10;
//  Tensor<int> a("a", {dim}, Format{Dense});
//  Tensor<int> b("b", {dim}, Format{Dense});
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");
//  a(i) = b(i);
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
//  stmt = stmt.distributeOnto({i}, {in}, {il}, a(i));
  stmt = stmt.distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j));
//  stmt = stmt.pushCommUnder(b(i), in);
  stmt = stmt.pushCommUnder(b(i, j), jn);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, summaMM) {
  int dim = 10;
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko");

  a(i, j) = b(i, k) * c(k, j);

  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto placement = GridPlacement({0, 1});
  auto placeA = a.partition(grid).place(grid, placement);
  auto placeB = b.partition(grid).place(grid, placement);
  auto placeC = c.partition(grid).place(grid, placement);
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<GEMM>();
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j))
      .split(k, ko, ki, 512)
      .reorder({ko, il, jl})
      .pushCommUnder(b(i, k), ko)
      .pushCommUnder(c(k, j), ko)
      .swapLeafKernel(il, gemm)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);

  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/summaMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cannonMM) {
  int dim = 10;
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense});

  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto placement = GridPlacement({0, 1});
  auto placeA = a.partition(grid).place(grid, placement);
  auto placeB = b.partition(grid).place(grid, placement);
  auto placeC = c.partition(grid).place(grid, placement);
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko"), kos("kos");
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<GEMM>();
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .divide(k, ko, ki, gx)
      .reorder({ko, i, j, ki})
      .distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j))
      // TODO (rohany): No clue how stagger works when it is over these...
//      .stagger(ko, {in, jn}, kos)
      .pushCommUnder(b(i, k), in)
      .pushCommUnder(c(k, j), in)
      // .pushCommUnder(b(i, k), kos)
      // .pushCommUnder(c(k, j), kos)
      .swapLeafKernel(il, gemm)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  // std::cout << stmt << std::endl;
  // std::cout << lowered << std::endl;
  // return;
  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/cannonMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cuda_cannonMM) {
  int dim = 10;
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense});

  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto placement = GridPlacement({0, 1});
  auto placeA = a.partition(grid).place(grid, placement, taco::ParallelUnit::DistributedGPU);
  auto placeB = b.partition(grid).place(grid, placement, taco::ParallelUnit::DistributedGPU);
  auto placeC = c.partition(grid).place(grid, placement, taco::ParallelUnit::DistributedGPU);
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko"), kos("kos");
  IndexVar iln("iln"), ill("ill"), jln("jln"), jll("jll"), kii("kii"), kio("kio"), kios("kios");
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<CuGEMM>();
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      // Schedule for each node.
      .distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j), taco::ParallelUnit::DistributedNode)
      .divide(k, ko, ki, gx)
      .reorder({ko, il, jl})
      .stagger(ko, {in, jn}, kos)
      .pushCommUnder(b(i, k), kos)
      .pushCommUnder(c(k, j), kos)
      // Schedule for each GPU within a node.
      .distribute({il, jl}, {iln, jln}, {ill, jll}, Grid(2, 2), taco::ParallelUnit::DistributedGPU)
      .divide(ki, kio, kii, 2)
      .reorder({kio, ill, jll})
      .stagger(kio, {iln, jln}, kios)
      .pushCommUnder(b(i, k), kios)
      .pushCommUnder(c(k, j), kios)
      .pushCommUnder(a(i, j), jln)
      .swapLeafKernel(ill, gemm)
      ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  ofstream f("../legion/cannonMM/taco-generated.cu");
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  f.close();
}

TEST(distributed, johnsonMM) {
  int dim = 10;
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense});

  // Each tensor lives on a different face of the processor cube.
  auto gx = ir::Var::make("gridDim", Int32, false, false, true);
  auto grid = Grid(gx, gx);
  auto cube = Grid(gx, gx, gx);
  auto placeA = a.partition(grid).place(cube, GridPlacement({0, 1, Face(0)}));
  auto placeB = b.partition(grid).place(cube, GridPlacement({0, Face(0), 1}));
  auto placeC = c.partition(grid).place(cube, GridPlacement({Face(0), 0, 1}));
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl");
  a(i, j) = b(i, k) * c(k, j);
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<GEMM>();
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, cube)
      .pushCommUnder(a(i, j), kn)
      .pushCommUnder(b(i, k), kn)
      .pushCommUnder(c(k, j), kn)
      .swapLeafKernel(il, gemm)
      ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/johnsonMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, solomonikMM) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> c("c", {dim, dim}, Format{Dense, Dense});

  int procs = 64;
  int C = 2;
  int rpoc = sqrt(procs / C);
  int rpoc3 = sqrt(procs / (pow(C, 3)));

  // All tensors are distributed onto the i-j face of the process cube.
  Grid partGrid = Grid(rpoc, rpoc);
  Grid procGrid = Grid(rpoc, rpoc, C);
  auto placeA = a.partition(partGrid).place(procGrid, GridPlacement({0, 1, Face(0)}));
  auto placeB = b.partition(partGrid).place(procGrid, GridPlacement({0, 1, Face(0)}));
  auto placeC = c.partition(partGrid).place(procGrid, GridPlacement({0, 1, Face(0)}));
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl"), k1("k1"), k2("k2"), k1s("k1s");
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  // To schedule for solomonik's algorithm, we'll distribute over i, j, k according to the
  // processor grid. Then, we divide the kl loop into k1 and k2 so that each partition of C
  // is operated on in chunks. Finally, we then stagger the k1 loop so that along each parallel
  // slice of k, a Cannon style shifting occurs.
  stmt = stmt
      .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, procGrid)
      .divide(kl, k1, k2, rpoc3)
      .reorder({k1, il, jl})
      .stagger(k1, {in, jn}, k1s)
      .pushCommUnder(a(i, j), k1s)
      .pushCommUnder(b(i, k), k1s)
      .pushCommUnder(c(k, j), k1s)
      ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  {
    ofstream f("../legion/solomonikMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, staggerNoDist) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> expected("expected", {dim, dim}, Format{Dense, Dense});

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      b.insert({i, j}, 1);
      expected.insert({i, j}, 1);
    }
  }

  IndexVar i("i"), j("j"), js("js");
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.stagger(j, {i}, js);

  a.compile(stmt);
  a.evaluate();
  ASSERT_TRUE(equals(a, expected));
}

TEST(distributed, reduction) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> c("c", {dim}, Format{Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");

  a(i) = b(i, j) * c(j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distribute({i, j}, {in, jn}, {il, jl}, Grid(2, 2))
      .pushCommUnder(a(i), jn)
      .pushCommUnder(b(i, j), jn)
      .pushCommUnder(c(j), jn)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, packingPlacement) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  auto grid = Grid(4, 4);
  auto placeGrid = Grid(4, 4, 4);
  auto stmt = a.partition(grid).place(placeGrid, GridPlacement({0, 1, Face(0)}));
  auto lowered = lower(stmt, "placeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
  // Also write it into a file.
  {
    ofstream f("../legion/placement-test/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
    f.close();
  }
}

TEST(distributed, heirPlacement) {
  int dim = 10;
  auto toString = [](IndexStmt stmt) {
    std::stringstream ss;
    ss << stmt;
    return ss.str();
  };
  {
    // Simple partitioning of a vector onto a vector of processors.
    Tensor<int> a("a", {dim}, Format{Dense});
    auto grid = Grid(4);
    auto stmt = a.placeHierarchy({
      {grid, grid, GridPlacement({0}), ParallelUnit::DistributedNode},
      {grid, grid, GridPlacement({0}), ParallelUnit::DistributedNode},
    });
    ASSERT_EQ(toString(stmt), "suchthat(forall(in, forall(iln, forall(ill, place(a(i))), Distributed, ParallelReduction, transfers: transfer(a(i))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(i, in, il, 4) and divide(il, iln, ill, 4))");
  }
  {
    // Doubly partition a matrix into matrices on each sub-partition.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto stmt = a.placeHierarchy({
      {grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
      {grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
    });
    ASSERT_EQ(toString(stmt), "suchthat(forall(distFused, forall(distFused1, forall(ill, forall(jll, place(a(i,j)))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({in, jn}, reorder(in, jn)) and divide(il, iln, ill, 4) and divide(jl, jln, jll, 4) and multiFuse({iln, jln}, reorder(iln, jln)))");
  }
  {
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto g1 = Grid(4);
    auto g2 = Grid(4, 4);
    auto stmt = a.placeHierarchy({
        {g2, g2, GridPlacement({0, Replicate()}), ParallelUnit::DistributedNode},
        {g2, g2, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
    });
    ASSERT_EQ(toString(stmt), "suchthat(forall(distFused, forall(distFused1, forall(ill, forall(jl, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(k, kn, kl, 4) and multiFuse({in, kn}, reorder(in, kn)) and divide(il, iln, ill, 4) and divide(j, jn, jl, 4) and multiFuse({iln, jn}, reorder(iln, jn)))");
  }
  {
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto stmt = a.placeHierarchy({
        {grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
        {grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedGPU},
    });
    std::cout << stmt << std::endl;
    auto lowered = lower(stmt, "placeLegion", false, true);
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
  }
  {
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto pgrid = Grid(4, 4, 4);
    auto stmt = a.placeHierarchy({
        {grid, pgrid, GridPlacement({0, 1, Face(0)}), ParallelUnit::DistributedNode},
        {grid, pgrid, GridPlacement({0, 1, Face(0)}), ParallelUnit::DistributedGPU},
    });
    std::cout << stmt << std::endl;
    auto lowered = lower(stmt, "placeLegion", false, true);
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
  }
}

TEST(distributed, placement) {
  int dim = 10;

  auto toString = [](IndexStmt stmt) {
    std::stringstream ss;
    ss << stmt;
    return ss.str();
  };

  {
    // Simple partitioning of a vector onto a vector of processors.
    Tensor<int> a("a", {dim}, Format{Dense});
    auto grid = Grid(4);
    a.partition(grid);
    auto stmt = a.place(grid, GridPlacement({0}));
    ASSERT_EQ(toString(stmt), "suchthat(forall(in, forall(il, place(a(i))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(i, in, il, 4))");
  }
  {
    // Place a matrix onto a vector of processors.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    // TODO (rohany): Here is a good place to test partitioning the other dimension of the matrix.
    auto grid = Grid(4);
    auto placeGrid = Grid(4);
    a.partition(grid);
    auto stmt = a.place(placeGrid, GridPlacement({0}));
    ASSERT_EQ(toString(stmt), "suchthat(forall(in, forall(il, forall(j, place(a(i,j)))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4))");
    // TODO (rohany): It seems like doing GridPlacement({1}) mimics partitioning across the y axis.
  }
  {
    // Place a vector onto a grid in different ways.
    Tensor<int> a("a", {dim}, Format{Dense});
    auto grid = Grid(4);
    auto placeGrid = Grid(4, 4);
    a.partition(grid);
    // Place the vector so that each row of the processor grid holds the chunk of the vector.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, Replicate()}))), "suchthat(forall(distFused, forall(il, forall(jl, place(a(i)))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({in, jn}, reorder(in, jn)))");
    // Place the vector so that each column of the processor grid holds the chunk of the vector.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 0}))), "suchthat(forall(distFused, forall(jl, forall(il, place(a(i)))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(j, jn, jl, 4) and divide(i, in, il, 4) and multiFuse({jn, in}, reorder(jn, in)))");
  }
  {
    // Place a matrix onto a 3-dimensional grid in different ways.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto placeGrid = Grid(4, 4, 4);
    a.partition(grid);
    // Replicate the tensor along each dimension in turn.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, 1, Replicate()}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, Replicate(), 1}))), "suchthat(forall(distFused, forall(il, forall(kl, forall(jl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(k, kn, kl, 4) and divide(j, jn, jl, 4) and multiFuse({in, kn, jn}, reorder(in, kn, jn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 0, 1}))), "suchthat(forall(distFused, forall(kl, forall(il, forall(jl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(k, kn, kl, 4) and divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({kn, in, jn}, reorder(kn, in, jn)))");
    // Placing the tensor in different orientations (like put the columns along the first axis of the grid).
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({1, 0, Replicate()}))), "suchthat(forall(distFused, forall(jl, forall(il, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(j, jn, jl, 4) and divide(i, in, il, 4) and divide(k, kn, kl, 4) and multiFuse({jn, in, kn}, reorder(jn, in, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({1, Replicate(), 0}))), "suchthat(forall(distFused, forall(jl, forall(kl, forall(il, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and divide(i, in, il, 4) and multiFuse({jn, kn, in}, reorder(jn, kn, in)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 1, 0}))), "suchthat(forall(distFused, forall(kl, forall(jl, forall(il, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(k, kn, kl, 4) and divide(j, jn, jl, 4) and divide(i, in, il, 4) and multiFuse({kn, jn, in}, reorder(kn, jn, in)))");
  }
}

TEST(distributed, placement2) {
  int dim = 10;
//  Tensor<int> a("a", {dim}, Format{Dense});
//  auto grid = Grid(4);
//  auto placeGrid = Grid(4, 4);
//  a.partition(grid);
//  auto stmt = a.place(placeGrid, GridPlacement({0, Replicate()}));
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  auto grid = Grid(3, 3);
  auto placeGrid = Grid(3, 3, 3);
  a.partition(grid);
  // Replicate the tensor along each dimension in turn.
//  auto stmt = a.place(placeGrid, GridPlacement({0, Replicate(), 1}));
  auto stmt = a.place(placeGrid, GridPlacement({Replicate(), 0, 1}));
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, nesting) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j");
  a(i, j) = b(j, i);
  auto stmt = a.getAssignment().concretize();
  auto lowered = lower(stmt, "computeLegion", false, true);
  std::cout << lowered << std::endl;
}