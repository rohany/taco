#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t grid, int32_t gridy);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPart, int32_t gridX, int32_t gridY);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridX, int32_t gridY);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridX, int32_t gridY);
// void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition part, int32_t gx);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition part, LogicalPartition bPart, LogicalPartition cPart, int32_t gx);
void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gx") == 0) {
      gx = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gy") == 0) {
      gy = atoi(args.argv[++i]);
      continue;
    }
    // TODO (rohany): Add a flag to do the validation or not.
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (gx == -1) {
    std::cout << "Please provide a grid x size with -gx." << std::endl;
    return;
  }
  if (gy == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");
  // runtime->fill_field(ctx, A, A, FID_VAL, valType(0));
  // runtime->fill_field(ctx, B, B, FID_VAL, valType(1));
  // runtime->fill_field(ctx, C, C, FID_VAL, valType(1));

  auto apart = partitionLegionA(ctx, runtime, A, gx, gy);
  auto bpart = partitionLegionA(ctx, runtime, B, gx, gy);
  auto cpart = partitionLegionA(ctx, runtime, C, gx, gy);

  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, apart, 0);
    tacoFill<valType>(ctx, runtime, B, bpart, 1);
    tacoFill<valType>(ctx, runtime, C, cpart, 1);

    // Place the tensors.
//    auto part = placeLegionA(ctx, runtime, A, gx, gy);
//    auto bPart = placeLegionB(ctx, runtime, B, gx, gy);
//    auto cPart = placeLegionC(ctx, runtime, C, gx, gy);
//
    // placeLegionA(ctx, runtime, A, apart, gx, gy);
    // placeLegionA(ctx, runtime, B, bpart, gx, gy);
    // placeLegionA(ctx, runtime, C, cpart, gx, gy);
    placeLegionB(ctx, runtime, A, gx, gy);
    placeLegionB(ctx, runtime, B, gx, gy);
    placeLegionB(ctx, runtime, C, gx, gy);

    // initCuBLAS(ctx, runtime);

    // Compute on the tensors.
    benchmark(ctx, runtime, [&]() { computeLegion(ctx, runtime, A, B, C, apart, bpart, cpart, gx); });
  }


  // The result should be equal to 1.
  tacoValidate<valType>(ctx, runtime, A, apart, valType(n));
}

TACO_MAIN(valType)
