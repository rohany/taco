#include "cblas.h"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;

struct task_1Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_2Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_3Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_4Args {
  int32_t k;
};

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridX, int32_t gridY) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((a2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX - 1)) / gridX) + ((a1_dimension + (gridX - 1)) / gridX - 1)), (a1_dimension - 1)), TACO_MIN((jn * ((a2_dimension + (gridY - 1)) / gridY) + ((a2_dimension + (gridY - 1)) / gridY - 1)), (a2_dimension - 1)));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) aRect = aRect.make_empty();

    aColoring[(*itr)] = aRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND, 15210);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridX, int32_t gridY) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((b2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)), (b1_dimension - 1)), TACO_MIN((jn * ((b2_dimension + (gridY - 1)) / gridY) + ((b2_dimension + (gridY - 1)) / gridY - 1)), (b2_dimension - 1)));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) bRect = bRect.make_empty();

    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridX, int32_t gridY) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((c2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)), (c1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)), (c2_dimension - 1)));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) cRect = cRect.make_empty();

    cColoring[(*itr)] = cRect;
  }
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_3Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);
  AccessorROdouble2 b_vals(b, FID_VAL);
  AccessorROdouble2 c_vals(c, FID_VAL);
  AccessorRWdouble2 a_vals(a, FID_VAL);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  auto aPartitionBounds = runtime->get_index_space_domain(ctx, a_index_space);
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

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, LogicalPartition bPart, LogicalPartition cPart, int32_t gridX) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);
  DomainT<2> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(aPartition));

  // I have aPartition here. Need to create bPartition.
  // DomainPointColoring bHandColoring, cHandColoring;
  // DomainT<3> iterDomain(Rect<3>({0,0,0}, {domain.bounds.hi[0], domain.bounds.hi[1], gridX - 1}));
  // DomainT<2> bIKDomain(Rect<2>({0, 0}, {domain.bounds.hi[0], gridX}));
  // DomainT<2> cKJDomain(Rect<2>({0, 0}, {gridX, domain.bounds.hi[1]}));
  // for (PointInDomainIterator<3> itr(iterDomain); itr(); itr++) {
  //   int in = (*itr)[0];
  //   int jn = (*itr)[1];
  //   int ko = (*itr)[2];
  //   auto aPartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, aPartition, Point<2>(in, jn)).get_index_space());
  //   int64_t aPartitionBounds0lo = aPartitionBounds.lo()[0];
  //   int64_t aPartitionBounds0hi = aPartitionBounds.hi()[0];
  //   int64_t aPartitionBounds1lo = aPartitionBounds.lo()[1];
  //   int64_t aPartitionBounds1hi = aPartitionBounds.hi()[1];

  //   Point<2> bStart = Point<2>(aPartitionBounds0lo, (ko * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX));
  //   Point<2> bEnd = Point<2>(TACO_MIN(((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) - 1) + aPartitionBounds0lo),(b1_dimension - 1)), TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),(b2_dimension - 1)));
  //   Rect<2> bRect = Rect<2>(bStart, bEnd);
  //   // TODO (rohany): Generate bounds code here eventually.
  //   bHandColoring[Point<2>(in, ko)] = bRect;

  //   Point<2> cStart = Point<2>((ko * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX), aPartitionBounds1lo);
  //   Point<2> cEnd = Point<2>(TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),(c1_dimension - 1)), TACO_MIN(((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) - 1) + aPartitionBounds1lo),(c2_dimension - 1)));
  //   Rect<2> cRect = Rect<2>(cStart, cEnd);
  //   cHandColoring[Point<2>(ko, jn)] = cRect;
  // }
  // auto bPartition = runtime->create_index_partition(ctx, b_index_space, bIKDomain, bHandColoring, LEGION_DISJOINT_COMPLETE_KIND, 15213);
  // auto cPartition = runtime->create_index_partition(ctx, c_index_space, cKJDomain, cHandColoring, LEGION_DISJOINT_COMPLETE_KIND, 15251);

  // Need to make some projection functors...
  class BProjFunc : public ProjectionFunctor {
  public:
    BProjFunc(Runtime* runtime) : ProjectionFunctor(runtime) {}
    using ProjectionFunctor::project;
    LogicalRegion project(const Mappable *mappable, unsigned index,
                          LogicalPartition upper_bound,
                          const DomainPoint &point) override {
      auto task = mappable->as_task();
      assert(task != NULL);

      // Get out K from the functor.
      int k = ((int*)(task->args))[0];
      auto target = Point<2>(point[0], k);
      return runtime->get_logical_subregion_by_color(upper_bound, target);
    }
    LogicalRegion project(const Mappable *mappable, unsigned index,
                          LogicalRegion upper_bound,
                          const DomainPoint &point) override {
      // Shouldn't get here?
      assert(false);
      return LogicalRegion();
    }
    virtual unsigned get_depth() const { return 0; }
  };
  class CProjFunc : public ProjectionFunctor {
  public:
    CProjFunc(Runtime* runtime) : ProjectionFunctor(runtime) {}
    LogicalRegion project(const Mappable *mappable, unsigned index,
                          LogicalPartition upper_bound,
                          const DomainPoint &point) override {
      auto task = mappable->as_task();
      assert(task != NULL);

      // Get out K from the functor.
      int k = ((int*)(task->args))[0];
      auto target = Point<2>(k, point[1]);
      return runtime->get_logical_subregion_by_color(upper_bound, target);
    }
    LogicalRegion project(const Mappable *mappable, unsigned index,
                          LogicalRegion upper_bound,
                          const DomainPoint &point) override {
      // Shouldn't get here?
      assert(false);
      return LogicalRegion();
    }
    virtual unsigned get_depth() const { return 0; }
  };
  runtime->register_projection_functor(15210, new BProjFunc(runtime), true /* silence_warnings */);
  runtime->register_projection_functor(15213, new CProjFunc(runtime), true /* silence_warnings */);

  FutureMap fm;
  for (int32_t ko = 0; ko < gridX; ko++) {
    RegionRequirement aReq = RegionRequirement(aPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
    aReq.add_field(FID_VAL);
//    LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
//    RegionRequirement bReq = RegionRequirement(bLogicalPartition, 15210, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    RegionRequirement bReq = RegionRequirement(bPart, 15210, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
//    LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
//    RegionRequirement cReq = RegionRequirement(cLogicalPartition, 15213, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    RegionRequirement cReq = RegionRequirement(cPart, 15213, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    task_4Args taskArgsRaw;
    taskArgsRaw.k = ko;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
    IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    fm = runtime->execute_index_space(ctx, launcher);

  }
  fm.wait_all_results();
}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
