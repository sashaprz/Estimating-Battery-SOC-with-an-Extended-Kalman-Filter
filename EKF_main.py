#LIBRARIES
import numpy as np
import sympy
import sympy as sp
from sympy.stats import Normal
from sympy import And
from sympy import Eq
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import inspect

#google sheets API
sheet_name = 'OCV_data_GOOD'
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\Sasha\OneDrive\C_Projects\Simulator\psyched-upgrade-416312-3b5d91fc6159.json", scope)

gc = gspread.authorize(credentials)
client = gspread.authorize(credentials)
sheet = client.open(sheet_name).sheet1

#EQUATIONS
#f and g functions (for calculating matricies)
def f_function(x_k_minus_1, i_measured):

    n = 0.90 #representing eta, the efficiency factor. LFP batteries have an efficiency of 90-95%
    Q = 0.800 #capacity in Amp-hours
    dt = 1 #~1 second delta t, size of timestep

    # guess for x_k_minus_1 based on state space model
    result = (x_k_minus_1 - ((n * dt / Q ) * i_measured))
    #if isinstance(i_measured, (list, tuple)):
    #    i_measured = i_measured[0]

    x_k_minus = result 

    return x_k_minus

def OCV_function(x_k_minus): 

    function = 0.2218 + 0.2331 * x_k_minus - 0.0062 * x_k_minus**2 + 0.0001 * x_k_minus**3 - 0.0000 * x_k_minus**4

    return function

#Observation Model
#I am using a RC model to represent my battery. This is a type of equivalent circuit model (ECM). The resistance values need to be calculated at various SOCs, so this is defining functions to do that
#this is assuming a linear model, however I don't think that will be an issue as I am using the points at k and k - 1, so close that the trend might as well be linear
def calculate_R1_R0():#V_measured, x_k_minus, x_k_minus_1):
    #since R1 and R0 are interconnected (you require one to find the other), they must be solved together. Therefore must use guess and check
    #max_iterations = 1
    #tolerance = 1e-6

    #intial value for R0, based off measurement but doesn't account for variations while SOC decreases. actual measurement was 0.385 but i increased so that it was a more even split between two resistors
    #R0 = 2.385 
    #V_measured = float(V_measured)

    #for num in range(max_iterations):
    #    R1_new = (OCV_function(x_k_minus) - OCV_function(x_k_minus_1, V_measured) - (1 - x_k_minus + x_k_minus_1) * R0) / (x_k_minus - x_k_minus_1)
    #    R1 = R1_new
    #    x = None
    #    if (1 - x_k_minus_1) == 0:
    #        #to prevent it being divided by zero and erroring
    #        x = 1
    #    else:
    #        x = (1 - x_k_minus_1)
    #    R0_new = (OCV_function(x_k_minus_1) - R1 + (1 - x_k_minus_1) * R0) / x
    #    x = OCV_function(x_k_minus_1) - R1
    
    #V2 = R0_new *  0.14 #Vin divided by Imin
    #V2 = float(V2)
    #V1 = V_measured - V2 
    #R1 = V1 / 0.16 #Vin divided by Imax

    ##Change R1, R0
    R1 = 6.955
    R0 = 6.955
    return R1, R0

def g_function(x_k_minus, x_k_minus_1, V_measured):
    #take voltage measurement from user, used to compare + also calculate value
    R1, R0 = calculate_R1_R0() #old function signature: V_measured, x_k_minus, x_k_minus_1)
    
    #defining i_R1, the current across the R1 resistor. since it's using the R1 and R0 values already calculated at time k, I don't need to index them again. But they are different for each timestep. 
    R_total = R1 + R0
    #calculating current across full circuit; defining i
    i = float(V_measured) / R_total

    #assuming that voltage across R1 is the same as the voltage across the circuit, since they are connected in series.
    i_R1 = float(V_measured)/ R1

    #calculating predicted voltage measurement
    #y_hat_k = OCV_function(x_k_minus) - R1 * i_R1 - R0 * i
    return OCV_function(x_k_minus) - R1 * i_R1 - R0 * i

    #returns expected terminal voltage

#matricies
def calculate_A_matrix(x_k_minus, x_k_minus_1):
    #equation = f_function('x_k_minus', 'x_k_minus_1')
    #the equation to calculate A matrix is partial erivative of f function with respect to x_k_minus_1. so It is equivalent to x_k_minus, or predicted value at time k, over x_k_minus_1, estimate at previous timestep. 
    #because I am taking the derivative, taking the derivative of two values returns 0, which then serves as a zero multiplier throughout the rest of my code. 
    #in order to avoid this, I am calculating the slope (rise/run) instead of the derivative. Conceptually, they are the same, especially because I have such a large number of datapoints, therefore the error is negligeable. 
    result = x_k_minus - x_k_minus_1 / 1

    A_matrix = result * -1 #so it isn't negative

    return A_matrix

#def calculate_B_matrix(x_k_minus, x_k_minus_1):
#    f = f_function(x_k_minus, x_k_minus_1) + wk
#    B_matrix = sp.diff(f, wk)
#    return B_matrix


def calculate_C_matrix(x_k_minus, x_k_minus_1, V_measured, vk):
    # Define symbolic variable for g
   # g_symbolic = sp.symbols('g')
    # Define symbolic variable for x_k_minus
    x_k_minus_symbolic = sp.symbols('x_k_minus_symbolic')
    # Calculate g
    g = g_function(x_k_minus_symbolic, x_k_minus_1, V_measured) + vk
    # Calculate the derivative of g with respect to x_k_minus
    C_matrix = sp.diff(g, x_k_minus_symbolic)
    # Evaluate the derivative at the given value of x_k_minus
    C_matrix_value = C_matrix.evalf(subs={x_k_minus_symbolic: x_k_minus})
    return C_matrix_value

#def calculate_D_matrix(x_k_minus):
#    g = g_function(x_k_minus, k) + vk
#   D_matrix = sp.diff(g, vk)
#    return D_matrix

    
def plot_SOC(estimations, intervals): 
    plt.figure(figsize=(8, 6))
    plt.plot(intervals, estimations, marker='o', markersize=3, linestyle='-', label='SOC predictions')
    #plt.plot(soc_fit, ocv_fit, linestyle='--', label='Fitted Curve (Degree {})'.format(degree)) #
    #equation_curve = np.polyval(coeffs, soc_values)
    #plt.plot(estimations, equation_curve, linestyle='-', label='Polynomial Curve', color='red')

    plt.ylabel('State of Charge Prediction(%)')
    plt.xlabel('Interval Number')
    plt.title('SOC predictions')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


#EKF equations
def main ():
    #VARIABLES
    x_k_plus = 0.0
    x_k_minus =  0.0
    x_k_minus_1 = 1.00

    #error matrices and terms
    sigma_k_minus_1 = 1.00 #initalizing error covariance matrix as equivalent to identify matrix
    x_k_minus_1 =  100 #final guess for previous timestep
    vk_scalar = 0.0
    wk = 0 
    vk = 1
    
    #measured values
    i_measured = 0.0
    V_measured = 0.0

    #lists so i can plot data afterwards
    estimations = []
    intervals = []

    #for sake of temporary simplicity, I am setting vk = 1 and ignoring the effects of noise. 
    """
    # calculating variance of Hall Effect sensor noise. standard deviation = error x callibration factor. it is a gaussian distribution with a zero mean
    mean = 0
    std_dev = 0.05 * 2.76854928e-4
    vk = Normal("vk", mean, std_dev)

    # Sample a value from the normal distribution to get vk_scalar
    #vk_scalar = np.random.normal(mean, std_dev)
    vk_scalar = vk_scalar * std_dev + mean  # Adjust vk_scalar to match the mean and standard deviation of the distribution
    """
    #wk represents the process noise error. it has a zero mean, and is given by E(wk*wk)^T = R_k, where R_k is the covariance matrix with the process noise. this is basically any unaccounted error or discrepencies in the system dynamics. Everything is defined by only one dimension, therefore calculating a covariance matrix doesn't make sense. So I will set wk to zero, because of the zero mean 
    #ukf is represented by this term: -n * dt / Q * i. I am leaving it as a local variable within the f_function for simplicity. 
    #ukg is I think the resistance values (R1, R0), which I've defined within a function to calculate R1 and R0, as they differ at different SOC values

    for k in range(1, 100): #put to 17274 after it works
        #extract i value from sheet so it's not constant, need to change it to use dataset instead
        row_index = k + 1
        i_measured = float(sheet.cell(row_index, 3).value)
        V_measured = float(sheet.cell(row_index, 2).value) #from google spreadsheet, second column. change it to extract from dataset instead of sheet (the sheet is the same data curve was created off) likewrise for current measurements (find dataset)

        #calculate x_k_minus_1 (initial guess) with state space model
        x_k_minus = f_function(x_k_minus_1, i_measured)
    
        #calculating A and C matrices breforehand to decrease computational time
        C_matrix = calculate_C_matrix(x_k_minus, x_k_minus_1, V_measured, vk)
        A_matrix = calculate_A_matrix(x_k_minus, x_k_minus_1)

        #calculate X hat k minus term - same as above term i think
        #state_estimate = f_function(x_k_minus, x_k_minus_1)
        #state_estimate_matrix = sympy.Matrix([state_estimate])

        #calculate error covariance time update
        
        sigma_k_minus =  float(A_matrix * sigma_k_minus_1 * A_matrix)
        print("sigma_k_minus:", sigma_k_minus)

        #calculate kalman gain - same potential error as above
        #vk_scalar = sp.symbols('vk_scalar')
        # Convert vk_scalar to a SymPy matrix compatible with C_matrix for addition
        #vk_scalar = sp.Matrix([[vk_scalar]])  # Adjust the structure to match C_matrix for addition
        #C_matrix = sp.Matrix([[C_matrix]])
        Kalman_gain = sigma_k_minus * C_matrix * ((C_matrix * sigma_k_minus * C_matrix + vk_scalar)**-1)
        #yk = sp.Matrix([[float(V_measured)]])
        
        #state estimate measuremnet update based on kalman gain
        R1, R0 = calculate_R1_R0()
        i_R1 = V_measured/ R1

        #converting datatypes 
        #R1 = float(R1) if isinstance(R1, (list, tuple)) else R1
        #i_R1 = float(i_R1) if isinstance(i_R1, (list, tuple)) else i_R1
        #R0 = float(R0) if isinstance(R0, (list, tuple)) else R0
        #i_measured = float(i_measured)

        OCV_result = OCV_function(x_k_minus)
        result = OCV_result - R1 * i_R1 - R0 * i_measured
        #result is on the order of 50, likely because R1/R0 values are incorrect. 

        #state space model
        #x_future = f_function(x_k_minus) #not calling uk, as uk represents the eta term, which is inside the f_function. 
        yk = g_function(x_k_minus, x_k_minus_1, V_measured)
        
        #yk = float(yk[0])
        x_k_plus = x_k_minus + Kalman_gain * (yk - result) #(yk - g_function(state_estimate, k)) #may get error here because i'm not defining it in terms of x_values

        #1 represents an identify matrix, but I wrote it as 1 for datatype compatability
        #scaling up values to ensure precision when multiplying very small numbers
        Kalman_gain_large = Kalman_gain * 1e96
        C_matrix_large = C_matrix * 1e96
        sigma_k_minus_large = sigma_k_minus * 1e96

        #error covariance measurment update
        sigma_k_plus_large = (1e96 - Kalman_gain_large * C_matrix_large) * sigma_k_minus_large 
        sigma_k_plus = sigma_k_plus_large / 1e96
        #print("sigma_k_plus:", sigma_k_plus)

        #print("interval: {}".format(k))
        #print("x_k_plus: {}, sigma_k_plus: {}".format(x_k_plus, sigma_k_plus))

        k += 1

        estimations.append(x_k_plus)
        intervals.append(k)

        #updating variables to prep for next timestep
        x_k_minus_1 = x_k_plus
        sigma_k_minus_1 = sigma_k_plus
        sigma_k_plus = 1

    plot_SOC(estimations, intervals)

    return estimations, intervals
    
main()
