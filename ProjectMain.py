# Imports
import numpy as np
import scipy as sp
from scipy import linalg
import openpyxl
from openpyxl import load_workbook
from openpyxl import Workbook


# Main
def main():
    tolerance, input_path, output_path = ask_user()
    values, input_line, bus_size = initialize(input_path)
    g_matrix, b_matrix = admittance(input_line)
    print("Stop here")


# Asks the user for tolerance, import file path, and output file path
def ask_user():
    print("What is your desired tolerance (in MVA)?")
    a = input()
    print("What is your input file path?")
    b = input()
    print("What is your output file path?")
    c = input()
    return a, b, c


# Initializes value matrix using the input file
# pinj, qinj, v, theta, type
def initialize(input_path):
    wb = load_workbook(input_path)
    input_bus = wb['BusData']
    input_line = wb['LineData']
    bus_size = input_bus.max_row - 1
    values = np.zeros((bus_size, 4))
    for i in range(bus_size):
        values[i, 0] = int(input_bus.cell(row=(i+2), column=5).value) - int(input_bus.cell(row=(i+2), column=2).value)
        values[i, 1] = int(input_bus.cell(row=(i+2), column=3).value)
        values[i, 2] = int(input_bus.cell(row=(i+2), column=6).value)
        values[i, 3] = 0
    return values, input_line, bus_size


# Admittance matrix Y = G + jB
def admittance(input_line):
    line_size = input_line.max_row - 1
    y = 1j * np.zeros((line_size, line_size))
    b_temp = np.zeros(line_size)

    # Populate admittance matrix indices ji, stores shunt admittance
    for i in range(input_line.max_row - 1):
        print(int(input_line.cell(row=(i+2), column=1).value))
        line_from = int(input_line.cell(row=(i+2), column=1).value)  # Line starts at this bus
        line_to = int(input_line.cell(row=(i+2), column=2).value)  # Line ends at this bus
        line_r = float(input_line.cell(row=(i+2), column=3).value)  # Line resistance
        line_x = float(input_line.cell(row=(i+2), column=4).value)  # Line reactance
        y[line_from - 1][line_to - 1] = -(line_r + 1j * line_x) ** -1  # Yji
        y[line_to - 1][line_from - 1] = -(line_r + 1j * line_x) ** -1  # Yij
        b_temp[line_from - 1] += float(input_line.cell(row=(i+2), column=5).value)  # Shunt admittance for start
        b_temp[line_to - 1] += float(input_line.cell(row=(i+2), column=5).value)  # Shunt admittance for end

    # Populate admittance matrix indices ii
    for i in range(line_size):
        y[i][i] = -sum([y[i][x] for x in range(line_size)]) + 0.5j * b_temp[i]
    g_matrix = y.real
    b_matrix = y.imag
    return g_matrix, b_matrix


main()