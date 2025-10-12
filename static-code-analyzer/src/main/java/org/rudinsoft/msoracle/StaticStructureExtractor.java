package org.rudinsoft.msoracle;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.RecordDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedReferenceTypeDeclaration;
import com.github.javaparser.resolution.types.ResolvedReferenceType;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class StaticStructureExtractor {

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: java StaticStructureExtractor <src_directory>");
            return;
        }

        CombinedTypeSolver typeSolver = new CombinedTypeSolver();

// JDK symbols
        typeSolver.add(new ReflectionTypeSolver());

// Project source
        typeSolver.add(new JavaParserTypeSolver(new File(args[0])));
        //typeSolver.add(new JavaParserTypeSolver(new File("/Users/pascal/Dev/ai-knowledgebase/static-code-analyzer/target/test-classes")));

        typeSolver.add(new JarTypeSolver("/Users/pascal/Dev/ftgo-application/ftgo-order-service/build/ftgo-order-service.jar"));

// External libs (added via `libs/`)
        File libDir = new File("/Users/pascal/Dev/ftgo-application/ftgo-order-service/build/libs/");
        if (libDir.exists() && libDir.isDirectory()) {
            for (File jar : libDir.listFiles((dir, name) -> name.endsWith(".jar"))) {
                try {
                    typeSolver.add(new JarTypeSolver(jar));
                } catch (IOException e) {
                    System.err.println("⚠️ Failed to add jar to type solver: " + jar.getName());
                }
            }
        }

        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        ParserConfiguration config = new ParserConfiguration().setSymbolResolver(symbolSolver).setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_17);
        StaticJavaParser.setConfiguration(config);

        List<Node> nodes = new ArrayList<>();
        List<Edge> edges = new ArrayList<>();

        Set<String> nodeIds = new HashSet<>();
        Set<String> edgeKeys = new HashSet<>();

        // Track known Event classes (filled in PASS 1)
        Set<String> eventClassIds = new HashSet<>();

        // Add a node once by id
        Consumer<Node> addNodeOnce = (n) -> {
            if (n == null || n.id == null) return;
            if (nodeIds.add(n.id)) {
                nodes.add(n);
            }
        };

        // Add an edge once by (from|type|to)
        Consumer<Edge> addEdgeOnce = (e) -> {
            if (e == null || e.from == null || e.to == null || e.type == null) return;
            String key = e.from + "|" + e.type + "|" + e.to;
            if (edgeKeys.add(key)) {
                edges.add(e);
            }
        };

        // Guess the library bucket from a package name
        Function<String, String> libOf = (pkg) -> {
            if (pkg == null) return "unknown";
            if (pkg.startsWith("java.") || pkg.startsWith("javax.")) return "jdk";
            if (pkg.startsWith("org.springframework")) return "spring";
            if (pkg.startsWith("org.apache")) return "apache";
            if (pkg.startsWith("com.fasterxml.jackson")) return "jackson";
            return "external";
        };

        // Ensure an external stub node exists (class or method) and return its id (prefixed with ext://)
        BiFunction<String, String, String> ensureExternalStub = (fqn, kind) -> {
            if (fqn == null || fqn.isEmpty()) return null;
            String stubId = "ext://" + fqn;
            if (!nodeIds.contains(stubId)) {
                Node stub = new Node();
                stub.id = stubId;
                stub.type = kind; // "ExternalClass" or "ExternalMethod"
                int lastDot = fqn.lastIndexOf('.');
                stub.name = lastDot >= 0 ? fqn.substring(lastDot + 1) : fqn;
                String pkg = lastDot > 0 ? fqn.substring(0, lastDot) : "";
                // Mark as external via annotations list (non-breaking)
                stub.annotations.add("external:true");
                stub.annotations.add("library:" + libOf.apply(pkg));
                addNodeOnce.accept(stub);
            }
            return stubId;
        };

        // --- JavaBeans getter/setter helpers ---
        Function<String, String> decap = (s) -> {
            if (s == null || s.isEmpty()) return s;
            if (s.length() > 1 && Character.isUpperCase(s.charAt(0)) && Character.isUpperCase(s.charAt(1))) {
                // e.g., URL -> URL (leave as-is)
                return s;
            }
            return Character.toLowerCase(s.charAt(0)) + s.substring(1);
        };

        Function<String, String> propFromGetter = (mname) -> {
            if (mname == null) return null;
            if (mname.startsWith("get") && mname.length() > 3) return decap.apply(mname.substring(3));
            if (mname.startsWith("is")  && mname.length() > 2) return decap.apply(mname.substring(2));
            return null;
        };

        Function<String, String> propFromSetter = (mname) -> {
            if (mname == null) return null;
            if (mname.startsWith("set") && mname.length() > 3) return decap.apply(mname.substring(3));
            return null;
        };

        // --- Domain Event marker interfaces (transitive) ---
        java.util.Set<String> EVENT_MARKER_IFACES = new java.util.HashSet<>();
        // Eventuate marker for domain events (may be implemented indirectly)
        EVENT_MARKER_IFACES.add("io.eventuate.tram.events.common.DomainEvent");

        Path start = Paths.get(args[0]);
        //Path start = Paths.get("/Users/pascal/Dev/ai-knowledgebase/static-code-analyzer/src/test/java");

        // Collect all .java files up front so we can do two passes
        List<Path> javaFiles = new ArrayList<>();
        try (var stream = Files.walk(start)) {
            stream.filter(f -> f.toString().endsWith(".java")).forEach(javaFiles::add);
        }

        // ---------------------------
        // PASS 1: add Class/Attribute/Method nodes (and defines/has_attribute edges)
        // ---------------------------
        for (Path file : javaFiles) {
            try {
                CompilationUnit cu = StaticJavaParser.parse(file);

                cu.findAll(ClassOrInterfaceDeclaration.class).forEach(clazz -> {
                    Node node = new Node();
                    node.type = clazz.isInterface() ? "Interface" : "Class";
                    node.name = clazz.getNameAsString();
                    node.id = clazz.getFullyQualifiedName().orElse(node.name);
                    node.file = file.toString();
                    node.startLine = clazz.getBegin().map(p -> p.line).orElse(-1);
                    node.endLine = clazz.getEnd().map(p -> p.line).orElse(-1);
                    clazz.getAnnotations().forEach(a -> node.annotations.add(a.getNameAsString()));

                    // If this is the marker interface itself, annotate it for clarity
                    if (clazz.isInterface()) {
                        if (EVENT_MARKER_IFACES.contains(clazz.getFullyQualifiedName().orElse(clazz.getNameAsString()))) {
                            node.annotations.add("EventMarker");
                        }
                    }

                    // Detect event classes by checking transitive ancestors for the marker interface
                    try {
                        boolean implementsEvent = false;
                        try {
                            ResolvedReferenceTypeDeclaration decl = clazz.resolve();
                            // Walk all ancestors (interfaces & superclasses) and check their qualified names
                            for (ResolvedReferenceType anc : decl.getAllAncestors()) {
                                String qn = null;
                                try {
                                    qn = anc.getQualifiedName();
                                } catch (UnsupportedOperationException uoe) {
                                    ResolvedReferenceTypeDeclaration idecl = anc.getTypeDeclaration().orElse(null);
                                    if (idecl != null) qn = idecl.getQualifiedName();
                                }
                                if (qn != null && EVENT_MARKER_IFACES.contains(qn)) { implementsEvent = true; break; }
                            }
                            // Also consider the declaration itself if it's exactly the marker (unlikely for classes)
                            if (!implementsEvent) {
                                try {
                                    String selfQN = decl.getQualifiedName();
                                    if (EVENT_MARKER_IFACES.contains(selfQN)) implementsEvent = true;
                                } catch (UnsupportedOperationException uoe) { /* ignore */ }
                            }
                        } catch (Exception e) {
                            // Fallback: shallow check on directly implemented types if full resolve failed
                            implementsEvent = clazz.getImplementedTypes().stream().anyMatch(t -> {
                                try {
                                    ResolvedType rt = t.resolve();
                                    if (!rt.isReferenceType()) return false;
                                    ResolvedReferenceType rrt = rt.asReferenceType();
                                    String qn;
                                    try { qn = rrt.getQualifiedName(); }
                                    catch (UnsupportedOperationException uoe) {
                                        ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                        qn = (idecl != null) ? idecl.getQualifiedName() : null;
                                    }
                                    return qn != null && EVENT_MARKER_IFACES.contains(qn);
                                } catch (Exception ex) {
                                    String name = t.getNameWithScope();
                                    if (name == null || name.isEmpty()) name = t.getNameAsString();
                                    return EVENT_MARKER_IFACES.contains(name);
                                }
                            });
                        }
                        if (implementsEvent && !clazz.isInterface()) {
                            node.type = "Event";            // override type to Event for better viz/filtering
                            node.annotations.add("Event");  // tag for downstream processing
                        }
                        // Record Event class ID for use in PASS 2
                        if ("Event".equals(node.type)) {
                            eventClassIds.add(node.id);
                        }
                    } catch (Exception ignore) {}

                    // Attributes
                    clazz.getFields().forEach(f -> {
                        f.getVariables().forEach(v -> {
                            Node attrNode = new Node();
                            attrNode.type = "Attribute";
                            attrNode.name = v.getNameAsString();
                            attrNode.id = node.id + "." + v.getNameAsString();
                            attrNode.file = file.toString();
                            attrNode.startLine = f.getBegin().map(p -> p.line).orElse(-1);
                            attrNode.endLine = f.getEnd().map(p -> p.line).orElse(-1);
                            try { attrNode.attr_type = v.getType().resolve().describe(); }
                            catch (Exception ex) { attrNode.attr_type = v.getTypeAsString(); }
                            addNodeOnce.accept(attrNode);
                            addEdgeOnce.accept(new Edge(node.id, attrNode.id, "has_attribute"));
                        });
                    });

                    // Type-level: Interface -> Class implemented_by
                    clazz.getImplementedTypes().forEach(t -> {
                        try {
                            ResolvedType rt = t.resolve();
                            String ifaceFqn = null;
                            if (rt.isReferenceType()) {
                                ResolvedReferenceType rrt = rt.asReferenceType();
                                try {
                                    ifaceFqn = rrt.getQualifiedName();
                                } catch (UnsupportedOperationException uoe) {
                                    // Fallback for older JP versions
                                    ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                    if (idecl != null) ifaceFqn = idecl.getQualifiedName();
                                }
                            }
                            if (ifaceFqn == null || ifaceFqn.isEmpty()) {
                                // best-effort fallback (may not be fully qualified)
                                ifaceFqn = t.getNameWithScope();
                                if (ifaceFqn == null || ifaceFqn.isEmpty()) ifaceFqn = t.getNameAsString();
                            }
                            String ifaceId = ifaceFqn;
                            if (!nodeIds.contains(ifaceId)) {
                                // create an external stub for the interface if it's not part of this repo
                                String stubId = ensureExternalStub.apply(ifaceFqn, "ExternalClass");
                                if (stubId != null) ifaceId = stubId;
                            }
                            addEdgeOnce.accept(new Edge(ifaceId, node.id, "implemented_by"));
                        } catch (Exception ex) {
                            // Fallback: try simple name as last resort
                            String ifaceId = t.getNameAsString();
                            if (!nodeIds.contains(ifaceId)) {
                                String stubId = ensureExternalStub.apply(ifaceId, "ExternalClass");
                                if (stubId != null) ifaceId = stubId;
                            }
                            addEdgeOnce.accept(new Edge(ifaceId, node.id, "implemented_by"));
                        }
                    });

                    // Methods (create nodes only; no call edges yet)
                    clazz.getMethods().forEach(m -> {
                        String methodName = m.getNameAsString();
                        boolean isGetterOrSetter = clazz.getFields().stream()
                            .flatMap(f -> f.getVariables().stream())
                            .anyMatch(v -> {
                                String fieldName = v.getNameAsString();
                                return methodName.equalsIgnoreCase("get" + fieldName) ||
                                       methodName.equalsIgnoreCase("set" + fieldName);
                            });
                        if (isGetterOrSetter) return;

                        String params = m.getParameters().stream()
                            .map(p -> { try { return p.getType().resolve().describe(); } catch (Exception e) { return p.getType().asString(); } })
                            .collect(Collectors.joining(", "));
                        String paramSignature = "(" + params + ")";
                        String methodId = node.id + "." + methodName + paramSignature;

                        addEdgeOnce.accept(new Edge(node.id, methodId, "defines"));

                        Node methodNode = new Node();
                        methodNode.id = methodId;
                        methodNode.type = "Method";
                        methodNode.name = methodName + paramSignature;
                        methodNode.file = file.toString();
                        methodNode.startLine = m.getBegin().map(p -> p.line).orElse(-1);
                        methodNode.endLine = m.getEnd().map(p -> p.line).orElse(-1);
                        methodNode.return_type = m.getType().asString();
                        methodNode.parameters = m.getParameters().stream()
                            .map(p -> { try { return p.getType().resolve().describe() + " " + p.getNameAsString(); } catch (Exception e) { return p.getType().asString() + " " + p.getNameAsString(); } })
                            .collect(Collectors.toList());

                        addNodeOnce.accept(methodNode);

                        // Method-level: InterfaceMethod -> ClassMethod implemented_by
                        try {
                            // Collect param types for this class method (FQ type names)
                            java.util.List<String> paramTypes = m.getParameters().stream()
                                .map(p -> {
                                    try { return p.getType().resolve().describe(); }
                                    catch (Exception e) { return p.getType().asString(); }
                                })
                                .collect(java.util.stream.Collectors.toList());

                            // For each directly implemented interface, try to find a matching method
                            clazz.getImplementedTypes().forEach(t -> {
                                try {
                                    ResolvedType rt = t.resolve();
                                    if (!rt.isReferenceType()) return;
                                    ResolvedReferenceType rrt = rt.asReferenceType();
                                    ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                    if (idecl == null || !idecl.isInterface()) return;

                                    // Search declared methods on the interface for same name & param arity/types
                                    idecl.getDeclaredMethods().forEach(im -> {
                                        if (!im.getName().equals(methodName)) return;
                                        if (im.getNumberOfParams() != paramTypes.size()) return;
                                        boolean same = true;
                                        for (int i = 0; i < im.getNumberOfParams(); i++) {
                                            String it = im.getParam(i).getType().describe();
                                            if (!it.equals(paramTypes.get(i))) { same = false; break; }
                                        }
                                        if (same) {
                                            // Build interface method id
                                            String ifaceFqn;
                                            try {
                                                ifaceFqn = rrt.getQualifiedName();
                                            } catch (UnsupportedOperationException uoe) {
                                                ifaceFqn = idecl.getQualifiedName();
                                            }
                                            String ifaceSig = ifaceFqn + "." + methodName + "(" + String.join(", ", paramTypes) + ")";
                                            String fromId = ifaceSig;
                                            if (!nodeIds.contains(fromId)) {
                                                String stubId = ensureExternalStub.apply(ifaceSig, "ExternalMethod");
                                                if (stubId != null) fromId = stubId;
                                            }
                                            addEdgeOnce.accept(new Edge(fromId, methodId, "implemented_by"));
                                        }
                                    });
                                } catch (Exception ex) {
                                    // ignore resolution failures quietly
                                }
                            });
                        } catch (Exception ex) {
                            // no-op
                        }
                    });

                    addNodeOnce.accept(node);
                });

                // Process Records
                cu.findAll(RecordDeclaration.class).forEach(record -> {
                    Node node = new Node();
                    node.type = "Record";
                    node.name = record.getNameAsString();
                    node.id = record.getFullyQualifiedName().orElse(node.name);
                    node.file = file.toString();
                    node.startLine = record.getBegin().map(p -> p.line).orElse(-1);
                    node.endLine = record.getEnd().map(p -> p.line).orElse(-1);
                    record.getAnnotations().forEach(a -> node.annotations.add(a.getNameAsString()));

                    // Detect event records by checking transitive ancestors for the marker interface
                    try {
                        boolean implementsEvent = false;
                        try {
                            ResolvedReferenceTypeDeclaration decl = record.resolve();
                            // Walk all ancestors (interfaces & superclasses) and check their qualified names
                            for (ResolvedReferenceType anc : decl.getAllAncestors()) {
                                String qn = null;
                                try {
                                    qn = anc.getQualifiedName();
                                } catch (UnsupportedOperationException uoe) {
                                    ResolvedReferenceTypeDeclaration idecl = anc.getTypeDeclaration().orElse(null);
                                    if (idecl != null) qn = idecl.getQualifiedName();
                                }
                                if (qn != null && EVENT_MARKER_IFACES.contains(qn)) { implementsEvent = true; break; }
                            }
                        } catch (Exception e) {
                            // Fallback: shallow check on directly implemented types if full resolve failed
                            implementsEvent = record.getImplementedTypes().stream().anyMatch(t -> {
                                try {
                                    ResolvedType rt = t.resolve();
                                    if (!rt.isReferenceType()) return false;
                                    ResolvedReferenceType rrt = rt.asReferenceType();
                                    String qn;
                                    try { qn = rrt.getQualifiedName(); }
                                    catch (UnsupportedOperationException uoe) {
                                        ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                        qn = (idecl != null) ? idecl.getQualifiedName() : null;
                                    }
                                    return qn != null && EVENT_MARKER_IFACES.contains(qn);
                                } catch (Exception ex) {
                                    String name = t.getNameWithScope();
                                    if (name == null || name.isEmpty()) name = t.getNameAsString();
                                    return EVENT_MARKER_IFACES.contains(name);
                                }
                            });
                        }
                        if (implementsEvent) {
                            node.type = "Event";            // override type to Event for better viz/filtering
                            node.annotations.add("Event");  // tag for downstream processing
                        }
                        // Record Event class ID for use in PASS 2
                        if ("Event".equals(node.type)) {
                            eventClassIds.add(node.id);
                        }
                    } catch (Exception ignore) {}

                    // Record components (parameters) as Attributes
                    record.getParameters().forEach(param -> {
                        Node attrNode = new Node();
                        attrNode.type = "Attribute";
                        attrNode.name = param.getNameAsString();
                        attrNode.id = node.id + "." + param.getNameAsString();
                        attrNode.file = file.toString();
                        attrNode.startLine = param.getBegin().map(p -> p.line).orElse(-1);
                        attrNode.endLine = param.getEnd().map(p -> p.line).orElse(-1);
                        try { attrNode.attr_type = param.getType().resolve().describe(); }
                        catch (Exception ex) { attrNode.attr_type = param.getTypeAsString(); }
                        addNodeOnce.accept(attrNode);
                        addEdgeOnce.accept(new Edge(node.id, attrNode.id, "has_attribute"));
                    });

                    // Explicit fields in records (if any)
                    record.getFields().forEach(f -> {
                        f.getVariables().forEach(v -> {
                            Node attrNode = new Node();
                            attrNode.type = "Attribute";
                            attrNode.name = v.getNameAsString();
                            attrNode.id = node.id + "." + v.getNameAsString();
                            attrNode.file = file.toString();
                            attrNode.startLine = f.getBegin().map(p -> p.line).orElse(-1);
                            attrNode.endLine = f.getEnd().map(p -> p.line).orElse(-1);
                            try { attrNode.attr_type = v.getType().resolve().describe(); }
                            catch (Exception ex) { attrNode.attr_type = v.getTypeAsString(); }
                            addNodeOnce.accept(attrNode);
                            addEdgeOnce.accept(new Edge(node.id, attrNode.id, "has_attribute"));
                        });
                    });

                    // Type-level: Interface -> Record implemented_by
                    record.getImplementedTypes().forEach(t -> {
                        try {
                            ResolvedType rt = t.resolve();
                            String ifaceFqn = null;
                            if (rt.isReferenceType()) {
                                ResolvedReferenceType rrt = rt.asReferenceType();
                                try {
                                    ifaceFqn = rrt.getQualifiedName();
                                } catch (UnsupportedOperationException uoe) {
                                    // Fallback for older JP versions
                                    ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                    if (idecl != null) ifaceFqn = idecl.getQualifiedName();
                                }
                            }
                            if (ifaceFqn == null || ifaceFqn.isEmpty()) {
                                // best-effort fallback (may not be fully qualified)
                                ifaceFqn = t.getNameWithScope();
                                if (ifaceFqn == null || ifaceFqn.isEmpty()) ifaceFqn = t.getNameAsString();
                            }
                            String ifaceId = ifaceFqn;
                            if (!nodeIds.contains(ifaceId)) {
                                // create an external stub for the interface if it's not part of this repo
                                String stubId = ensureExternalStub.apply(ifaceFqn, "ExternalClass");
                                if (stubId != null) ifaceId = stubId;
                            }
                            addEdgeOnce.accept(new Edge(ifaceId, node.id, "implemented_by"));
                        } catch (Exception ex) {
                            // Fallback: try simple name as last resort
                            String ifaceId = t.getNameAsString();
                            if (!nodeIds.contains(ifaceId)) {
                                String stubId = ensureExternalStub.apply(ifaceId, "ExternalClass");
                                if (stubId != null) ifaceId = stubId;
                            }
                            addEdgeOnce.accept(new Edge(ifaceId, node.id, "implemented_by"));
                        }
                    });

                    // Methods (create nodes only; no call edges yet)
                    // Filter out auto-generated accessor methods for record components
                    Set<String> recordComponentNames = record.getParameters().stream()
                        .map(p -> p.getNameAsString())
                        .collect(Collectors.toSet());

                    record.getMethods().forEach(m -> {
                        String methodName = m.getNameAsString();

                        // Skip auto-generated record accessor methods (getters)
                        if (recordComponentNames.contains(methodName) && m.getParameters().isEmpty()) {
                            return;
                        }

                        // Skip JavaBeans-style getters/setters for explicit fields
                        boolean isGetterOrSetter = record.getFields().stream()
                            .flatMap(f -> f.getVariables().stream())
                            .anyMatch(v -> {
                                String fieldName = v.getNameAsString();
                                return methodName.equalsIgnoreCase("get" + fieldName) ||
                                       methodName.equalsIgnoreCase("set" + fieldName);
                            });
                        if (isGetterOrSetter) return;

                        String params = m.getParameters().stream()
                            .map(p -> { try { return p.getType().resolve().describe(); } catch (Exception e) { return p.getType().asString(); } })
                            .collect(Collectors.joining(", "));
                        String paramSignature = "(" + params + ")";
                        String methodId = node.id + "." + methodName + paramSignature;

                        addEdgeOnce.accept(new Edge(node.id, methodId, "defines"));

                        Node methodNode = new Node();
                        methodNode.id = methodId;
                        methodNode.type = "Method";
                        methodNode.name = methodName + paramSignature;
                        methodNode.file = file.toString();
                        methodNode.startLine = m.getBegin().map(p -> p.line).orElse(-1);
                        methodNode.endLine = m.getEnd().map(p -> p.line).orElse(-1);
                        methodNode.return_type = m.getType().asString();
                        methodNode.parameters = m.getParameters().stream()
                            .map(p -> { try { return p.getType().resolve().describe() + " " + p.getNameAsString(); } catch (Exception e) { return p.getType().asString() + " " + p.getNameAsString(); } })
                            .collect(Collectors.toList());

                        addNodeOnce.accept(methodNode);

                        // Method-level: InterfaceMethod -> RecordMethod implemented_by
                        try {
                            // Collect param types for this record method (FQ type names)
                            java.util.List<String> paramTypes = m.getParameters().stream()
                                .map(p -> {
                                    try { return p.getType().resolve().describe(); }
                                    catch (Exception e) { return p.getType().asString(); }
                                })
                                .collect(java.util.stream.Collectors.toList());

                            // For each directly implemented interface, try to find a matching method
                            record.getImplementedTypes().forEach(t -> {
                                try {
                                    ResolvedType rt = t.resolve();
                                    if (!rt.isReferenceType()) return;
                                    ResolvedReferenceType rrt = rt.asReferenceType();
                                    ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                    if (idecl == null || !idecl.isInterface()) return;

                                    // Search declared methods on the interface for same name & param arity/types
                                    idecl.getDeclaredMethods().forEach(im -> {
                                        if (!im.getName().equals(methodName)) return;
                                        if (im.getNumberOfParams() != paramTypes.size()) return;
                                        boolean same = true;
                                        for (int i = 0; i < im.getNumberOfParams(); i++) {
                                            String it = im.getParam(i).getType().describe();
                                            if (!it.equals(paramTypes.get(i))) { same = false; break; }
                                        }
                                        if (same) {
                                            // Build interface method id
                                            String ifaceFqn;
                                            try {
                                                ifaceFqn = rrt.getQualifiedName();
                                            } catch (UnsupportedOperationException uoe) {
                                                ifaceFqn = idecl.getQualifiedName();
                                            }
                                            String ifaceSig = ifaceFqn + "." + methodName + "(" + String.join(", ", paramTypes) + ")";
                                            String fromId = ifaceSig;
                                            if (!nodeIds.contains(fromId)) {
                                                String stubId = ensureExternalStub.apply(ifaceSig, "ExternalMethod");
                                                if (stubId != null) fromId = stubId;
                                            }
                                            addEdgeOnce.accept(new Edge(fromId, methodId, "implemented_by"));
                                        }
                                    });
                                } catch (Exception ex) {
                                    // ignore resolution failures quietly
                                }
                            });
                        } catch (Exception ex) {
                            // no-op
                        }
                    });

                    addNodeOnce.accept(node);
                });

            } catch (Exception e) {
                System.err.println("Failed to parse " + file + ": " + e.getMessage());
            }
        }

        // ---------------------------
        // PASS 2: add behavioral edges (calls, creates, publishes_event, reads/writes_attribute)
        // ---------------------------
        for (Path file : javaFiles) {
            try {
                CompilationUnit cu = StaticJavaParser.parse(file);

                cu.findAll(ClassOrInterfaceDeclaration.class).forEach(clazz -> {
                    String classFqn = clazz.getFullyQualifiedName().orElse(clazz.getNameAsString());

                    clazz.getMethods().forEach(m -> {
                        String methodName = m.getNameAsString();
                        String params = m.getParameters().stream()
                            .map(p -> { try { return p.getType().resolve().describe(); } catch (Exception e) { return p.getType().asString(); } })
                            .collect(Collectors.joining(", "));
                        String methodId = classFqn + "." + methodName + "(" + params + ")";

                        // JavaBeans reads/writes + regular calls/external (publishes_event heuristic removed)
                        m.findAll(MethodCallExpr.class).forEach(call -> {
                            // JavaBeans reads/writes + regular calls/external
                            try {
                                ResolvedMethodDeclaration resolved = call.resolve();
                                String methodName2 = resolved.getName();
                                String declClassFqn = resolved.getPackageName() + "." + resolved.getClassName();

                                boolean isGetter = ((methodName2.startsWith("get") || methodName2.startsWith("is")) && resolved.getNumberOfParams() == 0);
                                boolean isSetter = (methodName2.startsWith("set") && resolved.getNumberOfParams() == 1);
                                if (isGetter || isSetter) {
                                    String prop = isGetter ? propFromGetter.apply(methodName2) : propFromSetter.apply(methodName2);
                                    if (prop != null) {
                                        String attrId = declClassFqn + "." + prop;
                                        if (nodeIds.contains(attrId)) {
                                            addEdgeOnce.accept(new Edge(methodId, attrId, isSetter ? "writes_attribute" : "reads_attribute"));
                                            return; // skip generic calls edge
                                        }
                                    }
                                }

                                String p2 = IntStream.range(0, resolved.getNumberOfParams())
                                        .mapToObj(i -> resolved.getParam(i).getType().describe())
                                        .collect(Collectors.joining(", "));
                                String calleeFqnWithSig = declClassFqn + "." + methodName2 + "(" + p2 + ")";

                                String toId = calleeFqnWithSig;
                                if (!nodeIds.contains(toId)) {
                                    String stubId = ensureExternalStub.apply(calleeFqnWithSig, "ExternalMethod");
                                    if (stubId != null) toId = stubId;
                                }
                                addEdgeOnce.accept(new Edge(methodId, toId, "calls"));
                            } catch (Exception e) {
                                if (!call.getScope().isPresent()) {
                                    String simpleName = call.getNameAsString();
                                    String propGet = propFromGetter.apply(simpleName);
                                    String propSet = propFromSetter.apply(simpleName);
                                    if (propGet != null) {
                                        String attrId = classFqn + "." + propGet;
                                        if (nodeIds.contains(attrId)) { addEdgeOnce.accept(new Edge(methodId, attrId, "reads_attribute")); return; }
                                    } else if (propSet != null) {
                                        String attrId = classFqn + "." + propSet;
                                        if (nodeIds.contains(attrId)) { addEdgeOnce.accept(new Edge(methodId, attrId, "writes_attribute")); return; }
                                    }
                                    String p3 = call.getArguments().stream().map(Object::toString).collect(Collectors.joining(", "));
                                    String fallbackCalleeId = classFqn + "." + simpleName + "(" + p3 + ")";
                                    addEdgeOnce.accept(new Edge(methodId, fallbackCalleeId, "calls"));
                                } else {
                                    System.err.println("⚠️ Could not resolve method call: " + call + " in " + methodId + " - " + e.getMessage());
                                }
                            }
                        });

                        // Object creations: if the created type is an Event (transitively implements DomainEvent),
                        // record a publishes_event edge; otherwise record a creates edge.
                        m.findAll(ObjectCreationExpr.class).forEach(call -> {
                            try {
                                String targetFqn = null;
                                boolean isEvent = false;
                                try {
                                    // Try high-fidelity resolution first
                                    ResolvedReferenceType rrt = call.getType().resolve().asReferenceType();
                                    // FQN of the created type
                                    try {
                                        targetFqn = rrt.getQualifiedName();
                                    } catch (UnsupportedOperationException uoe) {
                                        ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                        if (idecl != null) targetFqn = idecl.getQualifiedName();
                                    }
                                    // Transitive check: does it (or any ancestor) match the marker iface?
                                    if (targetFqn != null) {
                                        // Quick path: if we already recorded it as an Event in PASS 1
                                        if (eventClassIds.contains(targetFqn)) {
                                            isEvent = true;
                                        } else {
                                            // Walk ancestors for safety when not part of project sources
                                            ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                            if (idecl != null) {
                                                for (ResolvedReferenceType anc : idecl.getAllAncestors()) {
                                                    String qn = null;
                                                    try { qn = anc.getQualifiedName(); }
                                                    catch (UnsupportedOperationException uoe2) {
                                                        ResolvedReferenceTypeDeclaration aDecl = anc.getTypeDeclaration().orElse(null);
                                                        if (aDecl != null) qn = aDecl.getQualifiedName();
                                                    }
                                                    if (qn != null && EVENT_MARKER_IFACES.contains(qn)) { isEvent = true; break; }
                                                }
                                            }
                                        }
                                    }
                                } catch (Exception ignore) {
                                    // Fallback: derive FQN from descriptor so we still get an ID
                                    var descriptor = call.getType().toDescriptor();
                                    targetFqn = descriptor.substring(1, descriptor.length() - 1).replace('/', '.');
                                    // If we can't resolve, see if we already know it as an Event from PASS 1
                                    if (eventClassIds.contains(targetFqn)) {
                                        isEvent = true;
                                    }
                                }

                                if (targetFqn == null || targetFqn.isEmpty()) return;

                                String toId = targetFqn;
                                if (!nodeIds.contains(toId)) {
                                    String stubId = ensureExternalStub.apply(targetFqn, "ExternalClass");
                                    if (stubId != null) toId = stubId;
                                }

                                if (isEvent) {
                                    addEdgeOnce.accept(new Edge(methodId, toId, "publishes_event"));
                                } else {
                                    addEdgeOnce.accept(new Edge(methodId, toId, "creates"));
                                }
                            } catch (Exception ex) { /* ignore noisy failures */ }
                        });
                    });
                });

                // Process Records behavioral edges
                cu.findAll(RecordDeclaration.class).forEach(record -> {
                    String recordFqn = record.getFullyQualifiedName().orElse(record.getNameAsString());

                    record.getMethods().forEach(m -> {
                        String methodName = m.getNameAsString();
                        String params = m.getParameters().stream()
                            .map(p -> { try { return p.getType().resolve().describe(); } catch (Exception e) { return p.getType().asString(); } })
                            .collect(Collectors.joining(", "));
                        String methodId = recordFqn + "." + methodName + "(" + params + ")";

                        // JavaBeans reads/writes + regular calls/external
                        m.findAll(MethodCallExpr.class).forEach(call -> {
                            try {
                                ResolvedMethodDeclaration resolved = call.resolve();
                                String methodName2 = resolved.getName();
                                String declClassFqn = resolved.getPackageName() + "." + resolved.getClassName();

                                boolean isGetter = ((methodName2.startsWith("get") || methodName2.startsWith("is")) && resolved.getNumberOfParams() == 0);
                                boolean isSetter = (methodName2.startsWith("set") && resolved.getNumberOfParams() == 1);
                                if (isGetter || isSetter) {
                                    String prop = isGetter ? propFromGetter.apply(methodName2) : propFromSetter.apply(methodName2);
                                    if (prop != null) {
                                        String attrId = declClassFqn + "." + prop;
                                        if (nodeIds.contains(attrId)) {
                                            addEdgeOnce.accept(new Edge(methodId, attrId, isSetter ? "writes_attribute" : "reads_attribute"));
                                            return; // skip generic calls edge
                                        }
                                    }
                                }

                                String p2 = IntStream.range(0, resolved.getNumberOfParams())
                                        .mapToObj(i -> resolved.getParam(i).getType().describe())
                                        .collect(Collectors.joining(", "));
                                String calleeFqnWithSig = declClassFqn + "." + methodName2 + "(" + p2 + ")";

                                String toId = calleeFqnWithSig;
                                if (!nodeIds.contains(toId)) {
                                    String stubId = ensureExternalStub.apply(calleeFqnWithSig, "ExternalMethod");
                                    if (stubId != null) toId = stubId;
                                }
                                addEdgeOnce.accept(new Edge(methodId, toId, "calls"));
                            } catch (Exception e) {
                                if (!call.getScope().isPresent()) {
                                    String simpleName = call.getNameAsString();
                                    String propGet = propFromGetter.apply(simpleName);
                                    String propSet = propFromSetter.apply(simpleName);
                                    if (propGet != null) {
                                        String attrId = recordFqn + "." + propGet;
                                        if (nodeIds.contains(attrId)) { addEdgeOnce.accept(new Edge(methodId, attrId, "reads_attribute")); return; }
                                    } else if (propSet != null) {
                                        String attrId = recordFqn + "." + propSet;
                                        if (nodeIds.contains(attrId)) { addEdgeOnce.accept(new Edge(methodId, attrId, "writes_attribute")); return; }
                                    }
                                    String p3 = call.getArguments().stream().map(Object::toString).collect(Collectors.joining(", "));
                                    String fallbackCalleeId = recordFqn + "." + simpleName + "(" + p3 + ")";
                                    addEdgeOnce.accept(new Edge(methodId, fallbackCalleeId, "calls"));
                                } else {
                                    System.err.println("⚠️ Could not resolve method call: " + call + " in " + methodId + " - " + e.getMessage());
                                }
                            }
                        });

                        // Object creations: if the created type is an Event (transitively implements DomainEvent),
                        // record a publishes_event edge; otherwise record a creates edge.
                        m.findAll(ObjectCreationExpr.class).forEach(call -> {
                            try {
                                String targetFqn = null;
                                boolean isEvent = false;
                                try {
                                    // Try high-fidelity resolution first
                                    ResolvedReferenceType rrt = call.getType().resolve().asReferenceType();
                                    // FQN of the created type
                                    try {
                                        targetFqn = rrt.getQualifiedName();
                                    } catch (UnsupportedOperationException uoe) {
                                        ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                        if (idecl != null) targetFqn = idecl.getQualifiedName();
                                    }
                                    // Transitive check: does it (or any ancestor) match the marker iface?
                                    if (targetFqn != null) {
                                        // Quick path: if we already recorded it as an Event in PASS 1
                                        if (eventClassIds.contains(targetFqn)) {
                                            isEvent = true;
                                        } else {
                                            // Walk ancestors for safety when not part of project sources
                                            ResolvedReferenceTypeDeclaration idecl = rrt.getTypeDeclaration().orElse(null);
                                            if (idecl != null) {
                                                for (ResolvedReferenceType anc : idecl.getAllAncestors()) {
                                                    String qn = null;
                                                    try { qn = anc.getQualifiedName(); }
                                                    catch (UnsupportedOperationException uoe2) {
                                                        ResolvedReferenceTypeDeclaration aDecl = anc.getTypeDeclaration().orElse(null);
                                                        if (aDecl != null) qn = aDecl.getQualifiedName();
                                                    }
                                                    if (qn != null && EVENT_MARKER_IFACES.contains(qn)) { isEvent = true; break; }
                                                }
                                            }
                                        }
                                    }
                                } catch (Exception ignore) {
                                    // Fallback: derive FQN from descriptor so we still get an ID
                                    var descriptor = call.getType().toDescriptor();
                                    targetFqn = descriptor.substring(1, descriptor.length() - 1).replace('/', '.');
                                    // If we can't resolve, see if we already know it as an Event from PASS 1
                                    if (eventClassIds.contains(targetFqn)) {
                                        isEvent = true;
                                    }
                                }

                                if (targetFqn == null || targetFqn.isEmpty()) return;

                                String toId = targetFqn;
                                if (!nodeIds.contains(toId)) {
                                    String stubId = ensureExternalStub.apply(targetFqn, "ExternalClass");
                                    if (stubId != null) toId = stubId;
                                }

                                if (isEvent) {
                                    addEdgeOnce.accept(new Edge(methodId, toId, "publishes_event"));
                                } else {
                                    addEdgeOnce.accept(new Edge(methodId, toId, "creates"));
                                }
                            } catch (Exception ex) { /* ignore noisy failures */ }
                        });
                    });
                });

            } catch (Exception e) {
                System.err.println("Failed to parse " + file + ": " + e.getMessage());
            }
        }

        // Output to JSON
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (FileWriter out = new FileWriter("graph_nodes.json")) {
            gson.toJson(nodes, out);
        }
        try (FileWriter out = new FileWriter("graph_edges.json")) {
            gson.toJson(edges, out);
        }

        System.out.println("✅ Extracted " + nodes.size() + " nodes.");
    }
}