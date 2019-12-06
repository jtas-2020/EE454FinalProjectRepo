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
    values, input_line, bus_size, tot_implicit = initialize(input_path)
    g_matrix, b_matrix = admittance(input_line)

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "BusResults"
    ws2 = wb.create_sheet(title="ConvergenceHistory")
    ws3 = wb.create_sheet(title="LineFlow")
    wb.save(output_path + '\\output_results.xlsx')

    converged, mismatches, max_p, max_q = mismatch(tolerance, g_matrix, b_matrix, values, bus_size, tot_implicit)
    mismatches_old = mismatches.copy()
    diverged = False
    iteration = 0
    while not converged and not diverged:
        j_matrix = build_jacobian(values, g_matrix, b_matrix, mismatch_size=len(mismatches))
        corrections = correct(j_matrix, mismatches)
        values = update(corrections, values)
        converged, mismatches, max_p, max_q = mismatch(tolerance, g_matrix, b_matrix, values, bus_size, tot_implicit)
        for i in range(len(mismatches)):
            if mismatches[i] >= mismatches_old[i]:
                diverged = True
        iteration += 1
        mismatches_old = mismatches.copy()
    if diverged:
        print("No solution found.")
    else:
        print("Solution found!")
        # TODO print output files


# Asks the user for tolerance, import file path, and output file path
def ask_user():
    print("What is your desired tolerance (in MVA)?")
    print("Eg. 0.1")
    #a = float(input()) / 100.0
    a = 0.1 / 100.0  # TODO: remove
    print("What is your input file path?")
    print("Eg. C:\\Users\\jtsch\\Desktop\\system_SampleInput.xlsx")
    #b = input()
    b = "C:\\Users\\jtsch\\Desktop\\system_SampleInput.xlsx"  # TODO: remove
    print("What is your output file path?")
    print("Eg. C:\\Users\\jtsch\\Desktop")
    #c = input()
    c = "C:\\Users\\jtsch\\Desktop"  # TODO: remove
    return a, b, c


# Initializes value matrix using the input file
def initialize(input_path):
    wb = load_workbook(input_path)
    input_bus = wb['BusData']
    input_line = wb['LineData']
    bus_size = input_bus.max_row - 1
    values = np.zeros((bus_size, 5))
    tot_implicit = 0
    for i in range(bus_size): # p_inj, q_inj, v, theta, type
        values[i, 0] = float(input_bus.cell(row=(i+2), column=5).value)-float(input_bus.cell(row=(i+2), column=2).value)
        values[i, 0] /= 100.0
        values[i, 1] = -1*float(input_bus.cell(row=(i+2), column=3).value) / 100.0
        values[i, 2] = float(input_bus.cell(row=(i+2), column=6).value)
        values[i, 3] = 0.0
        bus_type = input_bus.cell(row=(i+2), column=4).value
        if bus_type == 'S':
            values[i, 4] = 0.0  # Slack bus
        elif bus_type == 'G':
            values[i, 4] = 1.0  # PV bus
            tot_implicit += 1
        else:
            values[i, 4] = 2.0  # PQ bus
            tot_implicit += 2
    return values, input_line, bus_size, tot_implicit


# Admittance matrix Y = G + jB
def admittance(input_line):
    line_size = input_line.max_row - 1
    y = 1j * np.zeros((line_size, line_size))
    b_temp = np.zeros(line_size)

    # Populate admittance matrix indices ji, stores shunt admittance
    for i in range(input_line.max_row - 1):
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


# Calculates the mismatch, decides if converged, and which buses have largest mismatch for real and reactive power
def mismatch(tolerance, g_matrix, b_matrix, values, bus_size, tot_implicit):
    mismatches = np.zeros(tot_implicit)
    max_p = 0.0000
    max_q = 0.0000
    converged = True
    for k in range(bus_size):  # Calculate real power mismatch
        if values[k][4] == 1.0 or 2.0:  # PV or PQ bus, respectively
            p_sum = 0.0
            for i in range(bus_size):
                temp_var = g_matrix[k][i]*np.cos(values[k][3]-values[i][3]) + b_matrix[k][i]*np.sin(values[k][3]-values[i][3])
                p_sum += values[k][2]*values[i][2]*temp_var
            p_sum -= values[k][0]  # Subtracts the actual real power injected from the summation
            if p_sum > tolerance:
                converged = False
            if max_p < p_sum:
                max_p = p_sum
    for k in range(bus_size):  # Calculate reactive power mismatch
        if values[k][4] == 2.0:  # PQ bus
            q_sum = 0.0
            for i in range(bus_size):
                temp_var = g_matrix[k][i]*np.sin(values[k][3]-values[i][3]) - b_matrix[k][i]*np.cos(values[k][3]-values[i][3])
                q_sum += values[k][2]*values[i][2]*temp_var
            q_sum -= values[k][1]  # Subtracts the actual reactive power injected from the summation
            if q_sum > tolerance:
                converged = False
            if max_q < q_sum:
                max_q = q_sum
    return converged, mismatches, max_p, max_q


# Creates the Jacobian matrix for the current iteration
def build_jacobian(values, g_matrix, b_matrix, mismatch_size):
    j_matrix = np.zeros((mismatch_size, mismatch_size))

    return j_matrix


# Creates the corrections needed for this iteration
def correct(j_matrix, mismatches):
    inv_j = -1 * np.linalg.inv(j_matrix)
    corrections = np.matmul(inv_j, mismatches)
    return corrections


# Updates the values matrix for this iteration
def update(corrections, values):

    return values


main()