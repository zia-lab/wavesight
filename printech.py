#!/usr/bin/env python3

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.syntax import Syntax

console = Console()

def printer(message, tail=' ...'):
    console.print('%s%s' % (message, tail), style="blue", justify="left")

def rule(text='',color='bold blue'):
    if text:
        console.print(Rule(text, style=color))
    else:
        console.print(Rule(style=color))

def table(column_names, rows, title='', show_lines=False):
    table = Table(title=title, show_lines=show_lines)

    # Add columns
    for column in column_names:
        table.add_column(column)

    # Add rows
    for row in rows:
        table.add_row(*row)

    console.print(table, style="green")

def code_print(code, language='bash'):
    console.print(Syntax(code, language, theme='monokai', line_numbers=True))
