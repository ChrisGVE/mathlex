use mathlex::{
    parse, parse_latex, parse_with_config, BinaryOp, ContextSource, Direction, Expression,
    ExpressionMetadata, InequalityOp, IntegralBounds, MathConstant, MathType, ParseError,
    ParserConfig, Position, Span, UnaryOp,
};

#[test]
fn test_version() {
    assert_eq!(mathlex::VERSION, "0.2.0");
}

#[test]
fn test_parser_config_default() {
    let config = ParserConfig::default();
    assert!(config.implicit_multiplication);
}

#[test]
fn test_parser_config_custom() {
    let config = ParserConfig {
        implicit_multiplication: false,
        ..ParserConfig::default()
    };
    assert!(!config.implicit_multiplication);
}

#[test]
fn test_parser_config_equality() {
    let config1 = ParserConfig::default();
    let config2 = ParserConfig {
        implicit_multiplication: true,
        ..ParserConfig::default()
    };
    let config3 = ParserConfig {
        implicit_multiplication: false,
        ..ParserConfig::default()
    };

    assert_eq!(config1, config2);
    assert_ne!(config1, config3);
}

#[test]
fn test_parser_config_clone() {
    let config = ParserConfig::default();
    let cloned = config.clone();
    assert_eq!(config, cloned);
}

#[test]
fn test_parse_simple() {
    let expr = parse("2 + 3").unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn test_parse_latex_simple() {
    let expr = parse_latex(r"\frac{1}{2}").unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Div,
            ..
        }
    ));
}

#[test]
fn test_parse_with_config_default() {
    let config = ParserConfig::default();
    let expr = parse_with_config("sin(x) + 2", &config).unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn test_parse_with_config_custom() {
    let config = ParserConfig {
        implicit_multiplication: false,
        ..ParserConfig::default()
    };
    let expr = parse_with_config("2 + 3", &config).unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn test_all_type_exports() {
    // Ensure all required types are exported
    let _expr: Expression = Expression::Integer(42);
    let _op: BinaryOp = BinaryOp::Add;
    let _unary: UnaryOp = UnaryOp::Neg;
    let _const: MathConstant = MathConstant::Pi;
    let _dir: Direction = Direction::Both;
    let _ineq: InequalityOp = InequalityOp::Lt;
    let _bounds: IntegralBounds = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };
    let _pos: Position = Position::start();
    let _span: Span = Span::start();
    let _err: ParseError = ParseError::empty_expression(None);
    let _math_type: MathType = MathType::Scalar;
    let _ctx_src: ContextSource = ContextSource::Explicit;
    let _meta: ExpressionMetadata = ExpressionMetadata::default();
}
