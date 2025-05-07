"""
Visualization tools for alpha factors.
"""
from typing import Any
from dataclasses import fields

from graphviz import Digraph

from .base import Operation, AlphaFactor
from .operations import (
    InputFeature, TsRank, TsMin, TsMax,
    TsCorrPearson, TsCorrSpearman
)


def get_operation_style(op: Operation) -> dict[str, Any]:
    """Get visualization style for operation."""
    styles = {
        InputFeature: {
            'shape': 'oval',
            'style': 'filled',
            'fillcolor': '#E8F8F5',
            'fontname': 'Helvetica'
        },
        TsRank: {
            'shape': 'box',
            'style': 'filled',
            'fillcolor': '#D4E6F1',
            'fontname': 'Helvetica'
        },
        TsMin: {
            'shape': 'box',
            'style': 'filled',
            'fillcolor': '#D5F5E3',
            'fontname': 'Helvetica'
        },
        TsMax: {
            'shape': 'box',
            'style': 'filled',
            'fillcolor': '#FADBD8',
            'fontname': 'Helvetica'
        },
        TsCorrPearson: {
            'shape': 'box',
            'style': 'filled',
            'fillcolor': '#FCF3CF',
            'fontname': 'Helvetica'
        },
        TsCorrSpearman: {
            'shape': 'box',
            'style': 'filled',
            'fillcolor': '#F5EEF8',
            'fontname': 'Helvetica'
        }
    }
    return styles.get(type(op), {
        'shape': 'box',
        'fontname': 'Helvetica'
    })


def get_operation_label(op: Operation) -> str:
    """Get label for operation node."""
    if isinstance(op, InputFeature):
        return f"Input\\n{op.feature_name}"
    
    # Get operation name without 'Ts' prefix
    op_name = type(op).__name__[2:]
    
    # Add window size for operations that have it
    if hasattr(op, 'window'):
        return f"{op_name}\\nwindow={op.window}"
    
    return op_name


def get_child_operations(op: Operation) -> list[Operation]:
    """Get all child operations of an operation node."""
    children = []
    for field in fields(op):
        value = getattr(op, field.name)  # Use field.name to get the field name
        if isinstance(value, Operation):
            children.append(value)
        elif isinstance(value, (list, tuple)):
            children.extend(x for x in value if isinstance(x, Operation))
    return children


def visualize_alpha(
    alpha: AlphaFactor,
    filename: str | None = None,
    format: str = 'png',
    view: bool = True
) -> Digraph:
    """Visualize alpha factor as a tree using Graphviz.
    
    Args:
        - `alpha` (AlphaFactor): Alpha factor to visualize
        - `filename` (str | None): Output filename (without extension)
        - `format` (str): Output format ('png', 'pdf', 'svg', etc.)
        - `view` (bool): Whether to open the generated visualization
        
    Returns:
        - `Graphviz Digraph` object
    """
    # Create graph
    dot = Digraph(comment='Alpha Factor Tree')
    dot.attr(rankdir='TB')
    
    # Counter for unique node IDs
    node_counter = 0
    
    def add_operation(op: Operation) -> str:
        """Add operation node to graph and return its ID."""
        nonlocal node_counter
        node_id = f"node{node_counter}"
        node_counter += 1
        
        # Add node with style
        dot.node(node_id, get_operation_label(op), **get_operation_style(op))
        
        # Add edges to child operations
        for child in get_child_operations(op):
            child_id = add_operation(child)
            dot.edge(node_id, child_id)
        
        return node_id
    
    # Build tree
    add_operation(alpha.root)
    
    # Save and show
    if filename:
        dot.render(filename, format=format, view=view, cleanup=True)
    
    return dot


def print_alpha_tree(alpha: AlphaFactor) -> None:
    """Print alpha factor tree in text format.
    
    Args:
        - `alpha` (AlphaFactor): Alpha factor to print
    """
    def print_operation(op: Operation, indent: str = "") -> None:
        """Print operation node."""
        print(f"{indent}└─ {get_operation_label(op)}")
        
        next_indent = indent + "   "
        for child in get_child_operations(op):
            print_operation(child, next_indent)
    
    print("Alpha Factor Tree:")
    print_operation(alpha.root)


def get_mermaid_style(op: Operation) -> str:
    """Get Mermaid style for operation node.
    
    Args:
        - `op` (Operation): Operation to get style for
        
    Returns:
        - `str`: Mermaid style string
    """
    styles = {
        InputFeature: 'style Input fill:#E8F8F5',
        TsRank: 'style Rank fill:#D4E6F1',
        TsMin: 'style Min fill:#D5F5E3',
        TsMax: 'style Max fill:#FADBD8',
        TsCorrPearson: 'style CorrP fill:#FCF3CF',
        TsCorrSpearman: 'style CorrS fill:#F5EEF8'
    }
    return styles.get(type(op), '')


def get_mermaid_label(op: Operation) -> str:
    """Get Mermaid-compatible label for operation node.
    
    Args:
        - `op` (Operation): Operation to get label for
        
    Returns:
        - `str`: Label string formatted for Mermaid
    """
    if isinstance(op, InputFeature):
        return f"Input<br/>{op.feature_name}"
    
    # Get operation name without 'Ts' prefix
    op_name = type(op).__name__[2:]
    
    # Add window size for operations that have it
    if hasattr(op, 'window'):
        return f"{op_name}<br/>window={op.window}"
    
    return op_name


def visualize_alpha_mermaid(alpha: AlphaFactor) -> str:
    """Generate Mermaid diagram code for alpha factor visualization.
    
    Args:
        - `alpha` (AlphaFactor): Alpha factor to visualize
        
    Returns:
        - `str`: Mermaid diagram code as string
    """
    # Start Mermaid graph
    mermaid = ["graph TD"]
    
    # Counter for unique node IDs
    node_counter = 0
    
    def add_operation(op: Operation) -> str:
        """Add operation node to Mermaid code and return its ID."""
        nonlocal node_counter, mermaid
        node_id = f"node{node_counter}"
        node_counter += 1
        
        # Add node with label
        label = get_mermaid_label(op)
        mermaid.append(f"    {node_id}[\"{label}\"]")
        
        # Add style if defined
        style = get_mermaid_style(op)
        if style:
            mermaid.append(f"    {style}")
        
        # Add edges to child operations
        for child in get_child_operations(op):
            child_id = add_operation(child)
            mermaid.append(f"    {node_id} --> {child_id}")
        
        return node_id
    
    # Build tree
    add_operation(alpha.root)
    
    return "\n".join(mermaid)


def save_mermaid_diagram(
    mermaid_code: str,
    filename: str
) -> None:
    """Save Mermaid diagram code to a file.
    
    Args:
        - `mermaid_code` (str): Mermaid diagram code
        - `filename` (str): Output filename (without extension)
    """
    with open(f"{filename}.mmd", "w") as f:
        f.write(mermaid_code) 