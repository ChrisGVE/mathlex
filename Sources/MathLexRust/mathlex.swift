public func parseText<GenericToRustStr: ToRustStr>(_ input: GenericToRustStr) throws -> Expression {
    return input.toRustStr({ inputAsRustStr in
        try { let val = __swift_bridge__$parse_text(inputAsRustStr); if val.is_ok { return Expression(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func parseLatex<GenericToRustStr: ToRustStr>(_ input: GenericToRustStr) throws -> Expression {
    return input.toRustStr({ inputAsRustStr in
        try { let val = __swift_bridge__$parse_latex_ffi(inputAsRustStr); if val.is_ok { return Expression(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func toString(_ expr: ExpressionRef) -> RustString {
    RustString(ptr: __swift_bridge__$expression_to_string(expr.ptr))
}
public func toLatex(_ expr: ExpressionRef) -> RustString {
    RustString(ptr: __swift_bridge__$expression_to_latex(expr.ptr))
}
public func findVariables(_ expr: ExpressionRef) -> RustVec<RustString> {
    RustVec(ptr: __swift_bridge__$expression_find_variables(expr.ptr))
}
public func findFunctions(_ expr: ExpressionRef) -> RustVec<RustString> {
    RustVec(ptr: __swift_bridge__$expression_find_functions(expr.ptr))
}
public func depth(_ expr: ExpressionRef) -> UInt {
    __swift_bridge__$expression_depth(expr.ptr)
}
public func nodeCount(_ expr: ExpressionRef) -> UInt {
    __swift_bridge__$expression_node_count(expr.ptr)
}

public class Expression: ExpressionRefMut {
    var isOwned: Bool = true

    public override init(ptr: UnsafeMutableRawPointer) {
        super.init(ptr: ptr)
    }

    deinit {
        if isOwned {
            __swift_bridge__$Expression$_free(ptr)
        }
    }
}
public class ExpressionRefMut: ExpressionRef {
    public override init(ptr: UnsafeMutableRawPointer) {
        super.init(ptr: ptr)
    }
}
public class ExpressionRef {
    var ptr: UnsafeMutableRawPointer

    public init(ptr: UnsafeMutableRawPointer) {
        self.ptr = ptr
    }
}
extension Expression: Vectorizable {
    public static func vecOfSelfNew() -> UnsafeMutableRawPointer {
        __swift_bridge__$Vec_Expression$new()
    }

    public static func vecOfSelfFree(vecPtr: UnsafeMutableRawPointer) {
        __swift_bridge__$Vec_Expression$drop(vecPtr)
    }

    public static func vecOfSelfPush(vecPtr: UnsafeMutableRawPointer, value: Expression) {
        __swift_bridge__$Vec_Expression$push(vecPtr, {value.isOwned = false; return value.ptr;}())
    }

    public static func vecOfSelfPop(vecPtr: UnsafeMutableRawPointer) -> Optional<Self> {
        let pointer = __swift_bridge__$Vec_Expression$pop(vecPtr)
        if pointer == nil {
            return nil
        } else {
            return (Expression(ptr: pointer!) as! Self)
        }
    }

    public static func vecOfSelfGet(vecPtr: UnsafeMutableRawPointer, index: UInt) -> Optional<ExpressionRef> {
        let pointer = __swift_bridge__$Vec_Expression$get(vecPtr, index)
        if pointer == nil {
            return nil
        } else {
            return ExpressionRef(ptr: pointer!)
        }
    }

    public static func vecOfSelfGetMut(vecPtr: UnsafeMutableRawPointer, index: UInt) -> Optional<ExpressionRefMut> {
        let pointer = __swift_bridge__$Vec_Expression$get_mut(vecPtr, index)
        if pointer == nil {
            return nil
        } else {
            return ExpressionRefMut(ptr: pointer!)
        }
    }

    public static func vecOfSelfAsPtr(vecPtr: UnsafeMutableRawPointer) -> UnsafePointer<ExpressionRef> {
        UnsafePointer<ExpressionRef>(OpaquePointer(__swift_bridge__$Vec_Expression$as_ptr(vecPtr)))
    }

    public static func vecOfSelfLen(vecPtr: UnsafeMutableRawPointer) -> UInt {
        __swift_bridge__$Vec_Expression$len(vecPtr)
    }
}



