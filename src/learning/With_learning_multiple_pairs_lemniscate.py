import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

# Configuration parameters
num_pairs = 5
E_charge = 5000
E_min = 3600
r = 2  # radius for tracking circles
d_charge = 0.2  # charging distance threshold

# Positioning parameters for pairs
pair_positions = {
    0: {'Xc1': 6.5, 'Yc1': 13, 'Xc2': 6.5, 'Yc2': 8, 'x_cs': 6.5, 'y_cs': 10.5},
    1: {'Xc1': 0.5, 'Yc1': 13, 'Xc2': 0.5, 'Yc2': 8, 'x_cs': 0.5, 'y_cs': 10.5},
    2: {'Xc1': -5.5,  'Yc1': 13, 'Xc2': -5.5,  'Yc2': 8, 'x_cs': -5.5,  'y_cs': 10.5},
    3: {'Xc1': -11.5,  'Yc1': 13, 'Xc2': -11.5,  'Yc2': 8, 'x_cs': -11.5,  'y_cs': 10.5},
    4: {'Xc1': -17.5,  'Yc1': 13, 'Xc2': -17.5,  'Yc2': 8, 'x_cs': -17.5,  'y_cs': 10.5}
}


E_low_dict = {
    0: 3616,
    1: 3637,
    2: 3616,
    3: 3616,
    4: 3617
}


def generate_figure8_trajectory(center_x, center_y, radius=1.5, points=300):
    theta = np.linspace(0, 2*np.pi, points)
    x = center_x + radius * np.sin(theta)
    y = center_y + radius * np.sin(theta) * np.cos(theta)
    return x, y


def read_csv_files():
    """Read all CSV files and return combined data"""
    all_data = {}

    for i in range(num_pairs):
        if i == 0:
            filename = 'different_trajectory/first pair/results_with_learning_first_pair_DT.csv'
        elif i == 1:
            filename = 'different_trajectory/second pair/results_with_learning_second_pair_DT.csv'
        elif i == 2:
            filename = 'different_trajectory/third pair/results_with_learning_third_pair_DT.csv'
        elif i == 3:
            filename = 'different_trajectory/fourth pair/results_with_learning_fourth_pair_DT.csv'
        elif i == 4:
            filename = 'different_trajectory/fifth pair/results_with_learning_fifth_pair_DT.csv'

        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()

        # Convert string flags to booleans
        df['nci_flag'] = df['nci_flag'].astype(str).str.strip().str.lower() == 'true'
        df['ncd_flag'] = df['ncd_flag'].astype(str).str.strip().str.lower() == 'true'
        df['dwnc_flag'] = df['dwnc_flag'].astype(str).str.strip().str.lower() == 'true'

        all_data[i] = df
        print(f"Loaded {filename}: {len(df)} rows")

    return all_data


def setup_animation():
    """Set up the matplotlib animation"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-25, 15)
    ax.set_ylim(2, 18)
    ax.set_aspect('equal')
    # ax.grid(True, alpha=0.3)
    ax.set_xlabel('x position', fontsize=12)
    ax.set_ylabel('y position', fontsize=12)
    
    return fig, ax

def initialize_plot_elements(ax):
    """Initialize all plot elements"""
    plot_elements = {
        'intelligent_robots': [],
        'dumb_robots': [],
        'tracking_circles_i': [],
        'tracking_circles_d': [],
        'charging_circles': [],
        'energy_text_i': [],
        'energy_text_d': [],
        'charging_flag_text_i': [],
        'charging_flag_text_d': [],
        'pair_labels': [],
        'e_lower_labels': [],
        'charging_range_circles': []

    }
    
    for i in range(num_pairs):
        pos = pair_positions[i]
        
        # Robot markers
        ir, = ax.plot([], [], 'ko', markersize=6)
        dr, = ax.plot([], [], 'bo', markersize=6)
        
        x8_i, y8_i = generate_figure8_trajectory(pos['Xc1'], pos['Yc1'], radius=r)
        x8_d, y8_d = generate_figure8_trajectory(pos['Xc2'], pos['Yc2'], radius=r)

        circ_i, = ax.plot(x8_i, y8_i, linestyle='--', color='black', alpha=0.7)
        circ_d, = ax.plot(x8_d, y8_d, linestyle='--', color='blue', alpha=0.7)

        # Dotted range circle for d_charge
        range_circle = Circle((pos['x_cs'], pos['y_cs']), radius=d_charge,
                      edgecolor='green', fill=False, linewidth=1.5, alpha=0.6)

        
        ax.add_patch(circ_i)
        ax.add_patch(circ_d)
        # ax.add_patch(circ_cs)
        ax.add_patch(range_circle)

        # Text elements
        ei_text = ax.text(0, 0, '', fontsize=8, color='black', ha='center')
        nci_text = ax.text(0, 0, '', fontsize=8, color='black', ha='center')
        ed_text = ax.text(0, 0, '', fontsize=8, color='blue', ha='center')
        ncd_text = ax.text(0, 0, '', fontsize=8, color='blue', ha='center')
        dwnc_text = ax.text(0, 0, '', fontsize=8, color='blue', ha='center')
        
        # Pair labels
        label = ax.text(pos['Xc1'], pos['Yc1'] + r + 1.5, f'Pair {num_pairs - i}', 
                       ha='center', fontsize=12, fontweight='bold')
        e_text = ax.text(pos['Xc1'], pos['Yc1'] + r + 1, f'$E_{{\\mathrm{{low}}}}$: {E_low_dict[i]}',
                ha='center', fontsize=10)

        
        # Store elements
        plot_elements['intelligent_robots'].append(ir)
        plot_elements['dumb_robots'].append(dr)
        plot_elements['tracking_circles_i'].append(circ_i)
        plot_elements['tracking_circles_d'].append(circ_d)
        # plot_elements['charging_circles'].append(circ_cs)
        plot_elements['energy_text_i'].append(ei_text)
        plot_elements['energy_text_d'].append(ed_text)
        plot_elements['charging_flag_text_i'].append(nci_text)
        plot_elements['charging_flag_text_d'].append(ncd_text)
        plot_elements['pair_labels'].append(label)
        plot_elements['e_lower_labels'].append(e_text)
        plot_elements.setdefault('wind_flag_text_i', []).append(dwnc_text)
        plot_elements['charging_range_circles'].append(range_circle)
    
    # Global text elements
    plot_elements['time_text'] = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                                        fontweight='bold', fontsize=14)
    plot_elements['e_charge_text'] = ax.text(0.02, 0.55, f'$E_{{max}}$ = {E_charge}', 
                                            transform=ax.transAxes, fontsize=10, 
                                            color='green', fontweight='bold')
    plot_elements['e_min_text'] = ax.text(0.02, 0.50, f'$E_{{min}}$ = {E_min}', 
                                         transform=ax.transAxes, fontsize=10, 
                                         color='red', fontweight='bold')
    
    return plot_elements

def get_robot_color(needs_charging, at_charging_station, default_color):
    """Determine robot color based on charging status"""
    if needs_charging:
        return 'green' if at_charging_station else 'red'
    return default_color

def animate_frame(frame_idx, all_data, plot_elements, max_frames):
    """Animate a single frame"""
    frame_idx = min(frame_idx, len(all_data[0]) - 1)

    
    # Update time text
    current_time = all_data[0]['timestep'].iloc[frame_idx] if frame_idx < len(all_data[0]) else 0
    plot_elements['time_text'].set_text(f'Time = {current_time:.2f}')
    
    artists = []
    
    for pair_idx in range(num_pairs):
        data = all_data[pair_idx]

        xi = data['xi'].iloc[frame_idx]
        yi = data['yi'].iloc[frame_idx]
        xd = data['xd'].iloc[frame_idx]
        yd = data['yd'].iloc[frame_idx]

        pos = pair_positions[pair_idx]
        dist_to_cs_i = np.hypot(xi - pos['x_cs'], yi - pos['y_cs'])
        dist_to_cs_d = np.hypot(xd - pos['x_cs'], yd - pos['y_cs'])
        at_cs_i = dist_to_cs_i <= d_charge
        at_cs_d = dist_to_cs_d <= d_charge

        # Get current data
        row = data.iloc[frame_idx]

        # Set robot colors based on charging status
        i_color = get_robot_color(row['nci_flag'], at_cs_i, 'black')
        d_color = get_robot_color(row['ncd_flag'], at_cs_d, 'blue')
        
        plot_elements['intelligent_robots'][pair_idx].set_color(i_color)
        plot_elements['dumb_robots'][pair_idx].set_color(d_color)
        
        # Update positions
        plot_elements['intelligent_robots'][pair_idx].set_data([xi], [yi])
        plot_elements['dumb_robots'][pair_idx].set_data([xd], [yd])
        
        # Update text positions and content
        plot_elements['energy_text_i'][pair_idx].set_position((xi, yi + 0.4))
        plot_elements['charging_flag_text_i'][pair_idx].set_position((xi, yi + 0.15))

        plot_elements['energy_text_d'][pair_idx].set_position((xd, yd - 0.35))
        plot_elements['charging_flag_text_d'][pair_idx].set_position((xd, yd - 0.6))
        plot_elements['wind_flag_text_i'][pair_idx].set_position((xd, yd - 0.85))
        
        plot_elements['energy_text_i'][pair_idx].set_text(f'E: {row["e_i"]:.1f}')
        plot_elements['charging_flag_text_i'][pair_idx].set_text(f'$n_{{ch}}$: {row["nci_flag"]}')

        plot_elements['energy_text_d'][pair_idx].set_text(f'E: {row["e_d"]:.1f}')
        plot_elements['charging_flag_text_d'][pair_idx].set_text(f'$n_{{ch}}$: {row["ncd_flag"]}')
        plot_elements['wind_flag_text_i'][pair_idx].set_text(f'$w_{{ch}}$: {row["dwnc_flag"]}')
        
        # Add to artists list
        artists.extend([
            plot_elements['intelligent_robots'][pair_idx],
            plot_elements['dumb_robots'][pair_idx],
            plot_elements['energy_text_i'][pair_idx],
            plot_elements['charging_flag_text_i'][pair_idx],
            plot_elements['energy_text_d'][pair_idx],
            plot_elements['charging_flag_text_d'][pair_idx],
            plot_elements['wind_flag_text_i'][pair_idx]
        ])
    
    artists.extend([
        plot_elements['time_text'],
        plot_elements['e_charge_text'],
        plot_elements['e_min_text']
    ])
    artists.extend(plot_elements['pair_labels'])
    artists.extend(plot_elements['e_lower_labels'])
    
    return artists

def main():
    """Main function to run the animation"""
    print("Loading CSV files...")
    all_data = read_csv_files()
    
    print("Setting up animation...")
    fig, ax = setup_animation()
    plot_elements = initialize_plot_elements(ax)
    
    # Find maximum number of frames
    max_frames = max(len(data) for data in all_data.values())
    print(f"Animation will have {max_frames} frames")
    
    # Create animation
    def animate(frame):
        return animate_frame(frame, all_data, plot_elements, max_frames)
    
    def init():
        artists = []
        for i in range(num_pairs):
            plot_elements['intelligent_robots'][i].set_data([], [])
            plot_elements['dumb_robots'][i].set_data([], [])
            plot_elements['energy_text_i'][i].set_text('')
            plot_elements['charging_flag_text_i'][i].set_text('')
            plot_elements['energy_text_d'][i].set_text('')
            plot_elements['charging_flag_text_d'][i].set_text('')
            plot_elements['wind_flag_text_i'][i].set_text('')

            artists.extend([
                plot_elements['intelligent_robots'][i],
                plot_elements['dumb_robots'][i],
                plot_elements['energy_text_i'][i],
                plot_elements['charging_flag_text_i'][i],
                plot_elements['energy_text_d'][i],
                plot_elements['wind_flag_text_i'][i],
                plot_elements['charging_flag_text_d'][i]
            ])
        
        plot_elements['time_text'].set_text('')
        artists.extend([
            plot_elements['time_text'],
            plot_elements['e_charge_text'],
            plot_elements['e_min_text']
        ])
        artists.extend(plot_elements['pair_labels'])
        artists.extend(plot_elements['e_lower_labels'])
        
        return artists
    
    print("Starting animation...")
    ani = animation.FuncAnimation(
        fig, animate, frames=max_frames, init_func=init,
        interval=10, blit=True, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return ani

if __name__ == "__main__":
    ani = main()