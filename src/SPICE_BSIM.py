
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from ltspice import Ltspice

def read_bsim_data(csv_path):
    """Read BSIM3v3 model data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        required_cols = ['VGS', 'VDS', 'ID']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV file must contain columns: {required_cols}")
        
        # Clean data - remove rows with NaN values
        df = df.dropna(subset=required_cols)
        return df[required_cols]
    except Exception as e:
        print(f"Error reading BSIM CSV file: {e}")
        raise

def parse_ltspice_raw(raw_path):
    """Parse LTspice .raw file using ltspice package"""
    try:
        # Initialize LTspice object
        l = Ltspice(raw_path)
        l.parse()  # Parse the raw file
        
        # Get available variables
        print("Available variables in .raw file:", l.variables)
        
        # Check for required variables - using the exact names from your output
        vds_var = 'V(vds)'
        vgs_var = 'V(vgs)'
        id_var = 'I(VDS)'
        
        # Verify the variables exist
        if vds_var not in l.variables:
            raise ValueError(f"Missing VDS variable. Found: {l.variables}")
        if vgs_var not in l.variables:
            raise ValueError(f"Missing VGS variable. Found: {l.variables}")
        if id_var not in l.variables:
            raise ValueError(f"Missing ID variable. Found: {l.variables}")
        
        # Get the data
        vds = l.get_data(vds_var)
        vgs = l.get_data(vgs_var)
        id_current = l.get_data(id_var)
        
        # Create DataFrame
        data = pd.DataFrame({
            'VDS': vds,
            'VGS': vgs,
            'ID': abs(id_current)  # Take absolute value of current
        })
        
        # Remove rows where any critical value is NaN or zero
        data = data[(data['VDS'].notna()) & 
                   (data['VGS'].notna()) & 
                   (data['ID'].notna())]
        data = data[(data['VDS'] != 0) & 
                   (data['VGS'] != 0) & 
                   (data['ID'] != 0)]
        
        print(f"LTspice data sample:\n{data.head()}")
        return data
    except Exception as e:
        print(f"Error parsing LTspice .raw file: {e}")
        raise

def create_id_vds_plot(bsim_data, ltspice_data):
    """Create interactive ID vs VDS comparison plot"""
    try:
        fig = go.Figure()
        
        # Clean data - ensure no NaN values
        bsim_data = bsim_data.dropna()
        ltspice_data = ltspice_data.dropna()
        
        # Print some debug info
        print("\nBSIM VGS range:", bsim_data['VGS'].min(), "to", bsim_data['VGS'].max())
        print("LTspice VGS range:", ltspice_data['VGS'].min(), "to", ltspice_data['VGS'].max())
        
        # Find common VGS values in both datasets (rounded to 1 decimal place)
        bsim_vgs_values = np.unique(np.round(bsim_data['VGS'], 1))
        ltspice_vgs_values = np.unique(np.round(ltspice_data['VGS'], 1))
        common_vgs = np.intersect1d(bsim_vgs_values, ltspice_vgs_values)
        
        print("Common VGS values:", common_vgs)
        
        if len(common_vgs) == 0:
            # If no common values, plot whatever data we have
            print("Warning: No common VGS values found, plotting all available data")
            common_vgs = np.unique(np.concatenate([
                np.round(bsim_data['VGS'].unique(), 1),
                np.round(ltspice_data['VGS'].unique(), 1)
            ]))
        
        # Sort the common VGS values and select up to 5 for clarity
        common_vgs = np.sort(common_vgs)
        if len(common_vgs) > 5:
            common_vgs = common_vgs[::len(common_vgs)//5]
        
        print("Selected VGS values for plotting:", common_vgs)
        
        # Get max VGS for color scaling
        max_vgs = max(common_vgs) if len(common_vgs) > 0 else 1
        
        # Add traces for each VGS value
        for vgs in common_vgs:
            # Skip if VGS is zero or NaN
            if np.isnan(vgs) or vgs == 0:
                continue
                
            # Calculate color
            try:
                hue = int(120 + 120 * vgs / max_vgs)
                color = f'hsl({hue}, 50%, 50%)'
            except:
                color = 'blue'  # fallback color
                
            # BSIM data
            bsim_subset = bsim_data[np.isclose(bsim_data['VGS'], vgs, atol=0.5)].sort_values('VDS')
            if not bsim_subset.empty:
                fig.add_trace(go.Scatter(
                    x=bsim_subset['VDS'],
                    y=bsim_subset['ID'],
                    name=f'BSIM VGS={vgs:.1f}V',
                    line=dict(color=color, width=3),
                    mode='lines',
                    legendgroup=f'VGS_{vgs:.1f}'
                ))
            
            # LTspice data
            ltspice_subset = ltspice_data[np.isclose(ltspice_data['VGS'], vgs, atol=0.5)].sort_values('VDS')
            if not ltspice_subset.empty:
                fig.add_trace(go.Scatter(
                    x=ltspice_subset['VDS'],
                    y=ltspice_subset['ID'],
                    name=f'LTspice VGS={vgs:.1f}V',
                    line=dict(color=color, width=3, dash='dash'),
                    mode='lines',
                    legendgroup=f'VGS_{vgs:.1f}'
                ))
        
        # Update layout
        fig.update_layout(
            title='MOSFET ID vs VDS Characteristics: BSIM vs LTspice Comparison',
            xaxis_title='Drain-Source Voltage V<sub>DS</sub> (V)',
            yaxis_title='Drain Current I<sub>D</sub> (A)',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            height=600,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        print(f"Error creating ID vs VDS plot: {e}")
        raise

def generate_report(bsim_csv_path, ltspice_raw_path, output_html_path):
    """Generate the comparison report"""
    try:
        print("\nReading BSIM data...")
        bsim_data = read_bsim_data(bsim_csv_path)
        print(f"Found {len(bsim_data)} BSIM data points")
        
        print("\nParsing LTspice data...")
        ltspice_data = parse_ltspice_raw(ltspice_raw_path)
        print(f"Found {len(ltspice_data)} LTspice data points")
        
        print("\nCreating comparison plot...")
        fig = create_id_vds_plot(bsim_data, ltspice_data)
        
        print("\nGenerating HTML report...")
        fig.write_html(output_html_path, auto_open=True)
        print(f"\nReport successfully generated at: {os.path.abspath(output_html_path)}")
        
    except Exception as e:
        print(f"\nError generating report: {e}")

if __name__ == '__main__':
    # Configuration - update these paths
    config = {
        'bsim_csv_path': r'D:\WORKSPACE\BSIM-Python-Implementation\data\BSIM3v3_2.csv',
        'ltspice_raw_path': r'D:\WORKSPACE\BSIM-Python-Implementation\model\Test.raw',
        'output_html_path': r'D:\WORKSPACE\BSIM-Python-Implementation\data\id_vds_comparison.html'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config['output_html_path']), exist_ok=True)
    
    generate_report(
        config['bsim_csv_path'],
        config['ltspice_raw_path'],
        config['output_html_path']
    )