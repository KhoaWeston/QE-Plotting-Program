from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
import os
import copy


class SubCellData:
    batch_name = ''
    sample_name = ''
    cell_name = ''
    subcell_ID = ''
    wavelength_list = []
    QE_list = []
    Jsc_AM15G = 0
    Jsc_AM0 = 0
    Eg = 0


def main():
    # Program Variables
    directory = 'C:/Users/khoaw/OneDrive/Personal Documents/UT - REU/Data/QE Data/07242024'  # TO CHANGE
    flag_continue = False
    curr_subcell = SubCellData()
    past_subcell = SubCellData()

    # Create Plots folder in directory
    if not os.path.exists(directory + '/Plots'):
        os.makedirs(directory + '/Plots')

    for files in os.listdir(directory):
        if not files.startswith('QE'):
            continue

        # Get sample ID from file name (MAY HAVE TO CHANGE)
        file_name = files.split('_')
        # curr_subcell.batch_name = file_name[2]
        curr_subcell.sample_name = file_name[2]
        curr_subcell.cell_name = file_name[3][0]
        curr_subcell.subcell_ID = file_name[3][1]

        # Open QE Measurement files
        curr_subcell.wavelength_list, curr_subcell.QE_list = np.loadtxt(os.path.join(directory, files), delimiter='\t',
                                                                        skiprows=17, unpack=True, usecols=(0, 1))

        # Get Jsc parameters
        curr_subcell.Jsc_AM15G, curr_subcell.Jsc_AM0, curr_subcell.Eg = run_calculations(curr_subcell)

        # Plot the file
        plot_graph(curr_subcell)

        # Save the plot
        plt.savefig(directory + '/Plots/' + files + '.png', bbox_inches='tight')

        # Clear the plot
        plt.clf()

        # Check if both subcells need plotted on one file
        if flag_continue:
            plot_both(curr_subcell, past_subcell)
            plt.savefig(directory + '/Plots/' + files[:len(files) - 1] + '.png', bbox_inches='tight')
            plt.clf()
            flag_continue = False
        else:
            # Save the previous file's data to plot on next loop
            past_subcell = copy.copy(curr_subcell)
            flag_continue = True


def run_calculations(curr_subcell):
    # Constants
    h = 6.626e-34  # Planck's constant
    q = 1.602e-19  # Charge of an electron
    c = 3e8  # Speed of light

    # Get AM1.5G and AM0 irradiance values with corresponding wavelengths
    wavelength_am15, irradiance_am15 = np.loadtxt('am1.5g-PVLighthouse.txt', unpack=True)
    wavelength_am0, irradiance_am0 = np.loadtxt('am0-PVLighthouse.txt', unpack=True)

    # Variables initializations used for calculations
    integrand_AM15G = []
    integrand_AM0 = []
    Jsc_am15g_sum = 0
    Jsc_am0_sum = 0
    QE_der_list = []
    last_QE = 0

    for i, (wave_len, QE_wave_len) in enumerate(zip(curr_subcell.wavelength_list, curr_subcell.QE_list)):
        # Method 1 Jsc Calculations
        AM15G_irrad = irradiance_am15[np.where(wavelength_am15 == wave_len)[0][0]]  # Find corresponding H
        AM0_irrad = irradiance_am0[np.where(wavelength_am0 == wave_len)[0][0]]
        integrand_AM15G.append((QE_wave_len / 100) * (AM15G_irrad * wave_len))  # EQE(wavelen) * N(wavelen)
        integrand_AM0.append((QE_wave_len / 100) * (AM0_irrad * wave_len))

        # Method 2 Jsc Calculations
        if i == 0:
            Jsc_am15g_sum = Jsc_am15g_sum + (QE_wave_len / 100) * ((AM15G_irrad * wave_len) * 1e-13) * q / (h * c)
            Jsc_am0_sum = Jsc_am0_sum + (QE_wave_len / 100) * ((AM15G_irrad * wave_len) * 1e-13) * q / (h * c)
        else:
            Jsc_am15g_sum = Jsc_am15g_sum + (QE_wave_len / 100) * ((AM15G_irrad * wave_len) * 1e-13) * 10 * q / (h * c)
            Jsc_am0_sum = Jsc_am0_sum + (QE_wave_len / 100) * ((AM15G_irrad * wave_len) * 1e-13) * 10 * q / (h * c)

        # Find the QE differentials for Eg calculation
        if i != 0:
            QE_der_list.append(last_QE - QE_wave_len)
        last_QE = QE_wave_len

    # Integrate over wavelengths in meters and convert to mA/cm^2
    wave_met = [i * 1e-9 for i in curr_subcell.wavelength_list]
    Jsc_AM15G = np.trapz(integrand_AM15G, wave_met) * q / (h * c) / 10
    Jsc_AM0 = np.trapz(integrand_AM0, wave_met) * q / (h * c) / 10

    # Calculate band gap
    Eg = 1240 / (curr_subcell.wavelength_list[QE_der_list.index(max(QE_der_list))]+5.)

    return Jsc_AM15G, Jsc_AM0, Eg


def plot_graph(subcell):
    # Plot the data
    plt.plot(subcell.wavelength_list, subcell.QE_list, color='r' if subcell.subcell_ID == 'a' else 'b',
             linestyle='-', marker='.')

    # Add labels to the plot
    plt.title('QE of {} Sample {} Cell {} ({}-Eg layer)'.format(
        subcell.batch_name, subcell.sample_name, subcell.cell_name, 'wide' if subcell.subcell_ID == 'a' else 'narrow'))
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('QE [%]')
    plt.ylim(0, 100)

    # Add text to the plot of Jsc
    mid = len(subcell.wavelength_list) // 2
    res = (subcell.wavelength_list[mid] + subcell.wavelength_list[~mid]) / 2
    plt.text(res, 90, 'Jsc (AM1.5G) = {:.2f} mA/cm{}'.format(subcell.Jsc_AM15G, get_super('2')), fontsize=10)
    plt.text(res, 80, 'Jsc (AM0) = {:.2f} mA/cm{}'.format(subcell.Jsc_AM0, get_super('2')), fontsize=10)
    plt.text(res, 70, 'Eg = {:.2f} eV'.format(subcell.Eg), fontsize=10)


# Plots both QEs of wide- and narrow-Eg layers in a separate file
def plot_both(curr_subcell, past_subcell):
    # Plot the data from both files on same graph
    plt.plot(past_subcell.wavelength_list, past_subcell.QE_list, color='r', linestyle='-', marker='.')
    plt.plot(curr_subcell.wavelength_list, curr_subcell.QE_list, color='b', linestyle='--', marker='+')
    plt.ylim(0, 100)

    # Add labels to plot
    plt.title('QE of {} Sample {} Cell {}'.format(
        curr_subcell.batch_name, curr_subcell.sample_name, curr_subcell.cell_name))
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('QE [%]')

    # Add to the legend
    wid_Eg_label = 'wide-Eg ({:.2f} eV):\n Jsc (AM1.5G) = {:.2f} mA/cm{}\n Jsc (AM0) = {:.2f} mA/cm{}'.format(
        past_subcell.Eg, past_subcell.Jsc_AM15G, get_super('2'), past_subcell.Jsc_AM0, get_super('2'))
    nar_Eg_label = 'narrow-Eg ({:.2f} eV):\n Jsc (AM1.5G) = {:.2f} mA/cm{}\n Jsc (AM0) = {:.2f} mA/cm{}'.format(
        curr_subcell.Eg, curr_subcell.Jsc_AM15G, get_super('2'), curr_subcell.Jsc_AM0, get_super('2'))
    patch1 = mpatches.Patch(color='r', label=wid_Eg_label)
    patch2 = mpatches.Patch(color='b', label=nar_Eg_label)
    plt.legend(handles=[patch1, patch2], loc="lower left")

    # Calculate Jsc Mismatches
    Jsc_AM15G_mis = abs(curr_subcell.Jsc_AM15G - past_subcell.Jsc_AM15G) / \
        min(curr_subcell.Jsc_AM15G, past_subcell.Jsc_AM15G)
    Jsc_AM0_mis = abs(curr_subcell.Jsc_AM0 - past_subcell.Jsc_AM0) / min(curr_subcell.Jsc_AM0, past_subcell.Jsc_AM0)

    # Add text to the plot
    plt.text(700, 93, 'Jsc Mismatch (AM1.5G) = {:.2f} %'.format(Jsc_AM15G_mis * 100), fontsize=10)
    plt.text(700, 85, 'Jsc Mismatch (AM0) = {:.2f} %'.format(Jsc_AM0_mis * 100), fontsize=10)

    return


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


if __name__ == "__main__":
    main()
