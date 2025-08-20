package org.rudinsoft.msoracle;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class Node {
    String id;
    String type;
    String name;
    String file;
    int startLine;
    int endLine;
    List<String> annotations = new ArrayList<>();
    List<String> methods = new ArrayList<>();
    String attr_type;
    String return_type;
    List<String> parameters;
}
