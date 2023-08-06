import pandas as pd

def Piecewise_RUL(df, config:dict)->list:
    """
    Generate piecewise RUL (Remaining Useful Life) values based on the provided DataFrame and configuration.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        list: A list of piecewise RUL values.
    """
    piecewise = []
    for id in df['machineID'].unique():
        # locate the max RUL for each machineID
        max_rul=round(df.loc[(df['machineID'] == id) & (df['failed'] == 1)][['RUL']].mean()[0]*config['RUL_max_percentage'])
        for i in df.loc[(df['machineID'] == id) & (df['failed'] == 1)].index:  
            stop = int(df[df.index == i]['RUL']) # with each time the machine fails as a cycle
            knee_point=stop-max_rul # determine when the degradation occurred based on the max_rul determined prior
            i_ini = i - stop + 1

            cycle_list = df.loc[(df.index >= i_ini) & (df.index <= i)]['RUL'].values
            
            piecewise_rul = []
            for j in range(len(cycle_list)):
                if knee_point<=0:
                    piecewise_rul.append(stop-cycle_list[j])
                elif j < knee_point:
                    piecewise_rul.append(max_rul)
                else:
                    rest = piecewise_rul[-1]-1
                    piecewise_rul.append(rest)

            piecewise.extend(piecewise_rul)

    return piecewise