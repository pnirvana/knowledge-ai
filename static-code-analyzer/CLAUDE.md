# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a static code analyzer for Java projects that builds a knowledge graph of code structure and relationships. It parses Java source code using JavaParser with symbol resolution to extract classes, interfaces, methods, attributes, and their relationships (calls, implementations, attribute access, etc.). The output is a graph representation with nodes and edges exported as JSON files.

## Build and Run Commands

**Build the project:**
```bash
mvn clean compile
```

**Run the analyzer (legacy StaticStructureExtractor):**
```bash
mvn exec:java -Dexec.mainClass="org.rudinsoft.msoracle.StaticStructureExtractor" -Dexec.args="<src_directory>"
```

**Run the analyzer (new Analyzer with GraphBuilder):**
```bash
mvn exec:java -Dexec.mainClass="org.rudinsoft.msoracle.Analyzer" -Dexec.args="--source <src_directory> --lib <lib_directory>"
```

Multiple `--source` and `--lib` arguments can be provided.

## Architecture

### Core Components

**Graph Model (Node.java, Edge.java):**
- `Node`: Represents code entities (Class, Interface, Method, Attribute, Event, ExternalClass, ExternalMethod)
- `Edge`: Represents relationships (calls, defines, has_attribute, implemented_by, creates, publishes_event, reads_attribute, writes_attribute)

**Two Analyzer Implementations:**

1. **StaticStructureExtractor.java** (Legacy, monolithic):
   - Single-file implementation with hardcoded paths
   - Two-pass analysis: Pass 1 creates nodes, Pass 2 creates behavioral edges
   - Outputs `graph_nodes.json` and `graph_edges.json`

2. **Analyzer.java + GraphBuilder.java** (New, modular):
   - `Analyzer`: Command-line entry point with `--source` and `--lib` flags
   - `GraphBuilder`: Configurable builder pattern for graph construction
   - Uses visitor pattern for extensibility

### Visitor Pattern Architecture

The new architecture uses a **two-level visitor pattern**:

**Level 1: SourceFileVisitor**
- `SourceFileVisitor` interface: Visits Java source files (Path → processing)
- `ClassParser`: Main implementation that parses files and extracts ClassOrInterfaceDeclarations

**Level 2: ClassOrInterfaceVisitor**
- `ClassOrInterfaceVisitor` interface: Visits individual class/interface declarations
- `ClassNodeExtractor`: Extracts class/interface nodes
- `AttributeNodeExtractor`: Extracts field/attribute nodes and has_attribute edges
- More extractors can be added (e.g., MethodNodeExtractor, CallGraphExtractor)

**Supporting Classes:**
- `ClassElement`: Wrapper around ClassOrInterfaceDeclaration with helper methods
- `Result`: Accumulates nodes and edges during parsing

### Key Technical Details

**Symbol Resolution:**
- Uses JavaParser's `JavaSymbolSolver` with `CombinedTypeSolver`
- Resolves types from: JDK (ReflectionTypeSolver), project sources (JavaParserTypeSolver), external JARs (JarTypeSolver)
- Required for accurate type resolution, method signatures, and transitive relationships

**Event Detection:**
- Identifies domain events by checking if classes transitively implement marker interfaces (e.g., `io.eventuate.tram.events.common.DomainEvent`)
- Creates `publishes_event` edges when Event classes are instantiated

**External Dependencies:**
- Stubs created for external classes/methods not in the project (prefixed with `ext://`)
- Annotated with library bucket (jdk, spring, apache, jackson, external)

**JavaBeans Support:**
- Detects getter/setter patterns and creates reads_attribute/writes_attribute edges
- Filters out getter/setter methods from method nodes to reduce noise

## Adding New Extractors

To add a new visitor (e.g., to extract method call graphs):

1. Create a class implementing `ClassOrInterfaceVisitor`
2. Implement `visit(ClassElement clazz)` method
3. Use `result.addNode()` and `result.addEdge()` to populate the graph
4. Add your visitor to the list in `GraphBuilder.build()`:
   ```java
   visitors.add(new ClassParser(result, List.of(
       new ClassNodeExtractor(result),
       new AttributeNodeExtractor(result),
       new YourNewExtractor(result)  // Add here
   )));
   ```

## Dependencies

- JavaParser 3.26.2 (core + symbol-solver)
- Gson 2.12.1 (JSON serialization)
- Lombok 1.18.36 (code generation)
- Java 17

## Output Format

The analyzer produces two JSON files:
- `graph_nodes.json`: Array of Node objects
- `graph_edges.json`: Array of Edge objects

These can be imported into graph databases or visualization tools for analysis.