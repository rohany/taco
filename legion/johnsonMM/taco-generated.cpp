#include "cblas.h"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#include "task_ids.h"
#include "taco/version.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int32_t shardingID;
  int32_t dim0;
  int32_t dim1;
  int32_t dim2;
  int32_t gridDim;
};
struct task_2Args {
  int32_t dim0;
  int32_t dim1;
  int32_t dim2;
  int32_t gridDim;
};
struct task_3Args {
  int32_t dim0;
  int32_t dim1;
  int32_t dim2;
  int32_t gridDim;
};
struct task_4Args {
  int32_t gridDim;
};

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridDim = args->gridDim;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridDim) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((a2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridDim - 1)) / gridDim) + ((a1_dimension + (gridDim - 1)) / gridDim - 1)), (a1_dimension - 1)), TACO_MIN((jn * ((a2_dimension + (gridDim - 1)) / gridDim) + ((a2_dimension + (gridDim - 1)) / gridDim - 1)), (a2_dimension - 1)));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) aRect = aRect.make_empty();

    aColoring[(*itr)] = aRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_COMPUTE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  return aLogicalPartition;
}

void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aLogicalPartition, int32_t gridDim, bool reg = false) {
  // Make a 3 dimensional grid to place the partition onto.
  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), 0);
  DomainT<3> domain = DomainT<3>(Rect<3>(lowerBound, upperBound));
  class PlaceProjFunc : public ProjectionFunctor {
  public:
    PlaceProjFunc(Runtime* runtime) : ProjectionFunctor(runtime) {}
    using ProjectionFunctor::project;
    LogicalRegion project(LogicalPartition upper_bound,
      const DomainPoint &point, const Domain& launch_domain) override {
      auto target = Point<2>(point[0], point[1]);
      return runtime->get_logical_subregion_by_color(upper_bound, target);
    }
    virtual bool is_functional() const { return true; }
    virtual unsigned get_depth() const { return 0; }
  };
  if (reg) {
    runtime->register_projection_functor(15451, new PlaceProjFunc(runtime), true /* silence_warnings */);
  }
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 15451, READ_ONLY, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  std::vector<int> dims = {gridDim, gridDim, gridDim};
  registerPlacementShardingFunctor(ctx, runtime, shardingID(420), dims);
  task_1Args taskArgsRaw;
  // TODO (rohany): Add a sharding functor ID here.
  taskArgsRaw.shardingID = shardingID(420);
  taskArgsRaw.dim0 = gridDim;
  taskArgsRaw.dim1 = gridDim;
  taskArgsRaw.dim2 = gridDim;
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridDim = args->gridDim;


  int32_t in = getIndexPoint(task, 0);
  int32_t kn = getIndexPoint(task, 1);
  int32_t jn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridDim) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), 0, (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[2];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((b2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), (b1_dimension - 1)), TACO_MIN((jn * ((b2_dimension + (gridDim - 1)) / gridDim) + ((b2_dimension + (gridDim - 1)) / gridDim - 1)), (b2_dimension - 1)));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) bRect = bRect.make_empty();

    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_COMPUTE_KIND);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.dim0 = gridDim;
  taskArgsRaw.dim1 = gridDim;
  taskArgsRaw.dim2 = gridDim;
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  launcher.tag = TACOMapper::PLACEMENT;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridDim = args->gridDim;


  int32_t kn = getIndexPoint(task, 0);
  int32_t in = getIndexPoint(task, 1);
  int32_t jn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridDim) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[1];
    int32_t jn = (*itr)[2];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridDim - 1)) / gridDim) + ((c1_dimension + (gridDim - 1)) / gridDim - 1)), (c1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), (c2_dimension - 1)));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) cRect = cRect.make_empty();

    cColoring[(*itr)] = cRect;
  }
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_COMPUTE_KIND);
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_3Args taskArgsRaw;
  taskArgsRaw.dim0 = gridDim;
  taskArgsRaw.dim1 = gridDim;
  taskArgsRaw.dim2 = gridDim;
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  launcher.tag = TACOMapper::PLACEMENT;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t gridDim = args->gridDim;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);
  AccessorROdouble2 b_vals(b, FID_VAL);
  AccessorROdouble2 c_vals(c, FID_VAL);
  AccessorReducedouble2 a_vals(a, FID_VAL, LEGION_REDOP_SUM_FLOAT64);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    (1 + (bDomain.hi()[0] - bDomain.lo()[0])),
    (1 + (cDomain.hi()[1] - cDomain.lo()[1])),
    (1 + (cDomain.hi()[0] - cDomain.lo()[0])),
    1.00000000,
    b_vals.ptr(bDomain.lo()),
    (b_vals.accessor.strides[0] / sizeof(double)),
    c_vals.ptr(cDomain.lo()),
    (c_vals.accessor.strides[0] / sizeof(double)),
    1.00000000,
    a_vals.ptr(aDomain.lo()),
    (a_vals.accessor.strides[0] / sizeof(double))
  );
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gridDim) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), (a1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), (a2_dimension - 1)));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) aRect = aRect.make_empty();

    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (kn * ((c1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), (b1_dimension - 1)), TACO_MIN((kn * ((c1_dimension + (gridDim - 1)) / gridDim) + ((c1_dimension + (gridDim - 1)) / gridDim - 1)), (b2_dimension - 1)));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) bRect = bRect.make_empty();

    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((kn * ((c1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> cEnd = Point<2>(TACO_MIN((kn * ((c1_dimension + (gridDim - 1)) / gridDim) + ((c1_dimension + (gridDim - 1)) / gridDim - 1)), (c1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), (c2_dimension - 1)));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) cRect = cRect.make_empty();

    cColoring[(*itr)] = cRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_COMPUTE_KIND);
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_COMPUTE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_COMPUTE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(a));
  aReq.add_field(FID_VAL);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_4Args taskArgsRaw;
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

// void computeLegion(Context ctx, Runtime* runtime,
//                    LogicalRegion a, LogicalRegion b, LogicalRegion c,
//                    LogicalPartition aPart, LogicalPartition bPart, LogicalPartition cPart, int32_t gridDim) {
//   int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
//   int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
//   auto a_index_space = get_index_space(a);
//   int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
//   int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
//   auto b_index_space = get_index_space(b);
//   int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
//   int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
//   auto c_index_space = get_index_space(c);
// 
//   class AProjFunc : public ProjectionFunctor {
//   public:
//     AProjFunc(Runtime* runtime) : ProjectionFunctor(runtime) {}
//     using ProjectionFunctor::project;
//     LogicalRegion project(LogicalPartition upper_bound,
//                           const DomainPoint &point, const Domain& launch_domain) override {
//       auto target = Point<2>(point[0], point[1]);
//       return runtime->get_logical_subregion_by_color(upper_bound, target);
//     }
//     virtual bool is_functional() const { return true; }
//     virtual unsigned get_depth() const { return 0; }
//   };
//   class BProjFunc : public ProjectionFunctor {
//   public:
//     BProjFunc(Runtime* runtime) : ProjectionFunctor(runtime) {}
//     using ProjectionFunctor::project;
//     LogicalRegion project(LogicalPartition upper_bound,
//       const DomainPoint &point, const Domain& launch_domain) override {
//           auto target = Point<2>(point[0], point[2]);
//       return runtime->get_logical_subregion_by_color(upper_bound, target);
//     }
//     virtual bool is_functional() const { return true; }
//     virtual unsigned get_depth() const { return 0; }
//   };
//   class CProjFunc : public ProjectionFunctor {
//   public:
//     CProjFunc(Runtime* runtime) : ProjectionFunctor(runtime) {}
//     LogicalRegion project(LogicalPartition upper_bound,
//                           const DomainPoint &point, const Domain& launch_domain) override {
//       auto target = Point<2>(point[2], point[1]);
//       return runtime->get_logical_subregion_by_color(upper_bound, target);
//     }
//     virtual bool is_functional() const { return true; }
//     virtual unsigned get_depth() const { return 0; }
//   };
//   runtime->register_projection_functor(15210, new BProjFunc(runtime), true /* silence_warnings */);
//   runtime->register_projection_functor(15213, new CProjFunc(runtime), true /* silence_warnings */);
//   runtime->register_projection_functor(15251, new AProjFunc(runtime), true /* silence_warnings */);
// 
//   Point<3> lowerBound = Point<3>(0, 0, 0);
//   Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), (gridDim - 1));
//   auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
//   DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
// 
//   DomainT<2> partDim = DomainT<2>(Rect<2>({0, 0}, {gridDim - 1, gridDim - 1}));
// 
//   DomainPointColoring aColoring = DomainPointColoring();
//   DomainPointColoring bColoring = DomainPointColoring();
//   DomainPointColoring cColoring = DomainPointColoring();
//   for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
//     int32_t in = (*itr)[0];
//     int32_t jn = (*itr)[1];
//     int32_t kn = (*itr)[2];
//     Point<2> aStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
//     Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), (a1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), (a2_dimension - 1)));
//     Rect<2> aRect = Rect<2>(aStart, aEnd);
//     auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
//     if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) aRect = aRect.make_empty();
// 
//     aColoring[Point<2>(in, jn)] = aRect;
//     Point<2> bStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (kn * ((c1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
//     Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), (b1_dimension - 1)), TACO_MIN((kn * ((c1_dimension + (gridDim - 1)) / gridDim) + ((c1_dimension + (gridDim - 1)) / gridDim - 1)), (b2_dimension - 1)));
//     Rect<2> bRect = Rect<2>(bStart, bEnd);
//     auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
//     if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) bRect = bRect.make_empty();
// 
//     bColoring[Point<2>(in, kn)] = bRect;
//     Point<2> cStart = Point<2>((kn * ((c1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
//     Point<2> cEnd = Point<2>(TACO_MIN((kn * ((c1_dimension + (gridDim - 1)) / gridDim) + ((c1_dimension + (gridDim - 1)) / gridDim - 1)), (c1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), (c2_dimension - 1)));
//     Rect<2> cRect = Rect<2>(cStart, cEnd);
//     auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
//     if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) cRect = cRect.make_empty();
// 
//     cColoring[Point<2>(kn, jn)] = cRect;
//   }
// 
//   auto aPartition = runtime->create_index_partition(ctx, a_index_space, partDim, aColoring, LEGION_COMPUTE_KIND);
//   auto bPartition = runtime->create_index_partition(ctx, b_index_space, partDim, bColoring, LEGION_COMPUTE_KIND);
//   auto cPartition = runtime->create_index_partition(ctx, c_index_space, partDim, cColoring, LEGION_COMPUTE_KIND);
// 
//   // RegionRequirement aReq = RegionRequirement(aPart, 15251, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(a));
//   RegionRequirement aReq = RegionRequirement(runtime->get_logical_partition(ctx, get_logical_region(a), aPartition), 15251, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(a));
//   aReq.add_field(FID_VAL);
//   // RegionRequirement bReq = RegionRequirement(bPart, 15210, READ_ONLY, EXCLUSIVE, get_logical_region(b));
//   RegionRequirement bReq = RegionRequirement(runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), 15210, READ_ONLY, EXCLUSIVE, get_logical_region(b));
//   bReq.add_field(FID_VAL);
//   // RegionRequirement cReq = RegionRequirement(cPart, 15213, READ_ONLY, EXCLUSIVE, get_logical_region(c));
//   RegionRequirement cReq = RegionRequirement(runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), 15213, READ_ONLY, EXCLUSIVE, get_logical_region(c));
//   cReq.add_field(FID_VAL);
//   task_4Args taskArgsRaw;
//   taskArgsRaw.gridDim = gridDim;
//   TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
//   IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
//   launcher.add_region_requirement(aReq);
//   launcher.add_region_requirement(bReq);
//   launcher.add_region_requirement(cReq);
//   auto fm = runtime->execute_index_space(ctx, launcher);
//   fm.wait_all_results();
// 
// }
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    if (TACO_FEATURE_OPENMP) {
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    } else {
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    }
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    if (TACO_FEATURE_OPENMP) {
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    } else {
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    }
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    if (TACO_FEATURE_OPENMP) {
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    } else {
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    }
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    if (TACO_FEATURE_OPENMP) {
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    } else {
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    }
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
