// RUN: cgeist %s --function=* -S | FileCheck %s

struct meta {
    long long a;
    char dtype;
};

struct fin {
    struct meta f;
    char dtype;
} __attribute__((packed)) ;

long long run(struct meta m, char c);

void compute(struct fin f) {
    run(f.f, f.dtype);
}

// CHECK-LABEL:   func.func @compute(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           %[[VAL_1:.*]] = llvm.alloca {{.*}} x !llvm.struct<(i64, i8)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, i8)>, i8)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<(i64, i8)>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, i8)>, i8)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i8
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_1]] : !llvm.struct<(i64, i8)>, !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i8)>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_1]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i8)>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i8
// CHECK-NEXT:      %[[VAL_10:.*]] = call @run(%[[VAL_7]], %[[VAL_9]], %[[VAL_5]]) : (i64, i8, i8) -> i64
// CHECK-NEXT:      return
// CHECK-NEXT:    }

