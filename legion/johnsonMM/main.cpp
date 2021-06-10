#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridX);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPart, int32_t gridX, bool reg = false);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridX);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridX);
//void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c);
void computeLegion(Context ctx, Runtime* runtime,
                   LogicalRegion a, LogicalRegion b, LogicalRegion c,
                   LogicalPartition aPart, LogicalPartition bPart, LogicalPartition cPart, int32_t gridDim);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gd = -1;
  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gdim") == 0) {
      gd = atoi(args.argv[++i]);
      continue;
    }
    // TODO (rohany): Add a flag to do the validation or not.
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (gd == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");
  tacoFill<valType>(ctx, runtime, A, 0); tacoFill<valType>(ctx, runtime, B, 1); tacoFill<valType>(ctx, runtime, C, 1);

  // Place the tensors.
  auto apart = partitionLegionA(ctx, runtime, A, gd);
  auto bpart = partitionLegionA(ctx, runtime, B, gd);
  auto cpart = partitionLegionA(ctx, runtime, C, gd);

  placeLegionA(ctx, runtime, A, apart, gd, true);
  placeLegionA(ctx, runtime, B, bpart, gd);
  placeLegionA(ctx, runtime, C, cpart, gd);
//  auto bpart = placeLegionB(ctx, runtime, B, gd);
//  auto cpart = placeLegionC(ctx, runtime, C, gd);

  // Compute on the tensors.
  benchmark([&]() { computeLegion(ctx, runtime, A, B, C, apart, bpart, cpart, gd); });

  tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)
