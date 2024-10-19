import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import chardet
import klib

# Initialize session state variables
if 'steps' not in st.session_state:
    st.session_state['steps'] = []

if 'code_snippets' not in st.session_state:
    st.session_state['code_snippets'] = []

if 'object_storers' not in st.session_state:
    st.session_state['object_storers'] = []

if 'steps_code_mapping' not in st.session_state:
    st.session_state['steps_code_mapping'] = {}

if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
class Conversions:
    def __init__(self):
        self.data = st.session_state['dataset']
    
    def astype(self):
        st.subheader("Perform DataFrame.astype()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.astype()</h4>", unsafe_allow_html=True)
        
        with st.expander("Customize Parameters for astype()"):
            dtype = st.text_area("Enter dtype (e.g., {'column1': 'int'})", key='Conversions-astype-dtype', value="{'column1': 'int'}")
            copy = st.selectbox("Copy the data?", options=[True, False], index=0, key='Conversions-astype-copy')
            errors = st.selectbox("Handle errors", options=['raise', 'ignore'], index=0, key='Conversions-astype-errors')
        
            if st.checkbox("Apply astype()", key="Conversions-astype-apply"):
                try:
                    result = self.data.astype(eval(dtype), copy=copy, errors=errors)
                    st.write("Resulting DataFrame:", result)
                    st.text("Resultent Data Types")
                    st.dataframe(result.dtypes)
                    
                    # Update session state
                    st.session_state['steps'].append('astype')
                    st.session_state['code_snippets'].append(f"df.astype({dtype}, copy={copy}, errors='{errors}')")
                    st.session_state['object_storers'].append(result)
                    st.session_state['steps_code_mapping']['astype'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                    
                    if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-astype-radio') == 'Yes':
                        st.session_state['dataset'] = result
                except Exception as e:
                    st.error(f"Error: {e}")
    
    def convert_dtypes(self):
        st.subheader("Perform DataFrame.convert_dtypes()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.convert_dtypes()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for convert_dtypes()"):
            infer_objects = st.selectbox("Infer object dtypes?", options=[True, False], index=0, key='Conversions-convert_dtypes-infer_objects')
            convert_string = st.selectbox("Convert to string dtype?", options=[True, False], index=0, key='Conversions-convert_dtypes-convert_string')
            convert_integer = st.selectbox("Convert to integer dtype?", options=[True, False], index=0, key='Conversions-convert_dtypes-convert_integer')
            convert_boolean = st.selectbox("Convert to boolean dtype?", options=[True, False], index=0, key='Conversions-convert_dtypes-convert_boolean')
            convert_floating = st.selectbox("Convert to floating dtype?", options=[True, False], index=0, key='Conversions-convert_dtypes-convert_floating')
            dtype_backend = st.selectbox("Select dtype backend", options=['numpy_nullable', 'pyarrow'], index=0, key='Conversions-convert_dtypes-dtype_backend')
        
            if st.checkbox("Apply convert_dtypes()", key="Conversions-convert_dtypes-apply"):
                result = self.data.convert_dtypes(
                    infer_objects=infer_objects,
                    convert_string=convert_string,
                    convert_integer=convert_integer,
                    convert_boolean=convert_boolean,
                    convert_floating=convert_floating,
                    dtype_backend=dtype_backend
                )
                st.write("Resulting DataFrame:", result)
                st.text("Resultent Data Types")
                st.dataframe(result.dtypes)
                
                # Update session state
                st.session_state['steps'].append('convert_dtypes')
                st.session_state['code_snippets'].append(
                    f"df.convert_dtypes(infer_objects={infer_objects}, convert_string={convert_string}, convert_integer={convert_integer}, "
                    f"convert_boolean={convert_boolean}, convert_floating={convert_floating}, dtype_backend='{dtype_backend}')"
                )
                st.session_state['object_storers'].append(result)
                st.session_state['steps_code_mapping']['convert_dtypes'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                
                if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-convert_dtypes-radio') == 'Yes':
                    st.session_state['dataset'] = result

    def infer_objects(self):
        st.subheader("Perform DataFrame.infer_objects()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.infer_objects()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for infer_objects()"):
            copy = st.selectbox("Copy the data?", options=[True, False], index=0, key='Conversions-infer_objects-copy')
        
            if st.checkbox("Apply infer_objects()", key="Conversions-infer_objects-apply"):
                result = self.data.infer_objects(copy=copy)
                st.write("Resulting DataFrame:", result)
                st.text("Resultent Data Types")
                st.dataframe(result.dtypes)
                
                # Update session state
                st.session_state['steps'].append('infer_objects')
                st.session_state['code_snippets'].append(f"df.infer_objects(copy={copy})")
                st.session_state['object_storers'].append(result)
                st.session_state['steps_code_mapping']['infer_objects'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                
                if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-infer_objects-radio') == 'Yes':
                    st.session_state['dataset'] = result

    def copy(self):
        st.subheader("Perform DataFrame.copy()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.copy()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for copy()"):
            deep = st.selectbox("Deep copy?", options=[True, False], index=0, key='Conversions-copy-deep')
        
            if st.checkbox("Apply copy()", key="Conversions-copy-apply"):
                result = self.data.copy(deep=deep)
                st.write("Resulting DataFrame:", result)
                st.text("Resultent Data Types")
                st.dataframe(result.dtypes)
                
                # Update session state
                st.session_state['steps'].append('copy')
                st.session_state['code_snippets'].append(f"df.copy(deep={deep})")
                st.session_state['object_storers'].append(result)
                st.session_state['steps_code_mapping']['copy'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                
                if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-copy-radio') == 'Yes':
                    st.session_state['dataset'] = result

    def to_numpy(self):
        st.subheader("Perform DataFrame.to_numpy()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.to_numpy()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for to_numpy()"):
            dtype = st.text_area("Enter dtype (e.g., 'float')", key='Conversions-to_numpy-dtype', value='None')
            copy = st.selectbox("Copy the data?", options=[True, False], index=0, key='Conversions-to_numpy-copy')

            if st.checkbox("Apply to_numpy()", key="Conversions-to_numpy-apply"):
                result = self.data.to_numpy(dtype=None if dtype == 'None' else eval(dtype), copy=copy)
                st.write("Resulting NumPy array:", result)
                st.text("Resultent Data Types")
                st.dataframe(result.dtypes)

                # Update session state
                st.session_state['steps'].append('to_numpy')
                st.session_state['code_snippets'].append(f"df.to_numpy(dtype={dtype}, copy={copy})")
                st.session_state['object_storers'].append(pd.DataFrame(result))  # Store as DataFrame for compatibility
                st.session_state['steps_code_mapping']['to_numpy'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

                if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-to_numpy-radio') == 'Yes':
                    st.session_state['dataset'] = pd.DataFrame(result)

class FirstNRows:
    def __init__(self):
        # Initialize dataset from session state
        self.data = st.session_state['dataset']
    
    def head(self):
        st.subheader("Perform DataFrame.head()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.head()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for head()"):
            n = st.slider("Number of rows to display (n)", min_value=1, max_value=100, value=5, key='Conversions-head-n')

            if st.checkbox("Apply head()", key="Conversions-head-apply"):
                result = self.data.head(n)
                st.write("Resulting DataFrame:", result)

                # Append to steps, code_snippets, and object_storers
                st.session_state['steps'].append('head')
                st.session_state['code_snippets'].append(f"df.head(n={n})")
                st.session_state['object_storers'].append(result)

                # Update dictionary
                st.session_state['steps_code_mapping']['head'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

                if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-head-radio') == 'Yes':
                    st.session_state['dataset'] = result

    def tail(self):
        st.subheader("Perform DataFrame.tail()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.tail()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for tail()"):
            n = st.slider("Number of rows to display (n)", min_value=1, max_value=100, value=5, key='Conversions-tail-n')

            if st.checkbox("Apply tail()", key="Conversions-tail-apply"):
                result = self.data.tail(n)
                st.write("Resulting DataFrame:", result)

                # Append to steps, code_snippets, and object_storers
                st.session_state['steps'].append('tail')
                st.session_state['code_snippets'].append(f"df.tail(n={n})")
                st.session_state['object_storers'].append(result)

                # Update dictionary
                st.session_state['steps_code_mapping']['tail'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

                if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-tail-radio') == 'Yes':
                    st.session_state['dataset'] = result

    def take(self):
        st.subheader("Perform DataFrame.take()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.take()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for take()"):
            indices = st.text_area("Enter indices as a list (e.g., [0, 1, 2])", value="[0, 1, 2]", key='Conversions-take-indices')
            axis = st.selectbox("Axis", options=[0, 1], index=0, key='Conversions-take-axis')

            if st.checkbox("Apply take()", key="Conversions-take-apply"):
                try:
                    result = self.data.take(eval(indices), axis=axis)
                    st.write("Resulting DataFrame:", result)

                    # Append to steps, code_snippets, and object_storers
                    st.session_state['steps'].append('take')
                    st.session_state['code_snippets'].append(f"df.take(indices={indices}, axis={axis})")
                    st.session_state['object_storers'].append(result)

                    # Update dictionary
                    st.session_state['steps_code_mapping']['take'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

                    if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-take-radio') == 'Yes':
                        st.session_state['dataset'] = result
                except Exception as e:
                    st.error(f"Error: {e}")

    def truncate(self):
        st.subheader("Perform DataFrame.truncate()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.truncate()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for truncate()"):
            before = st.text_input("Truncate before (e.g., index or date)", value="None", key='Conversions-truncate-before')
            if before:
                if before.isdigit():
                    before=int(before)
                else:
                    before=str(before)
            after = st.text_input("Truncate after (e.g., index or date)", value="None", key='Conversions-truncate-after')
            if after:
                if after.isnumeric():
                    after=int(after)
                else:
                    after=str(after)
            axis = st.selectbox("Axis", options=[0, 1], index=0, key='Conversions-truncate-axis')
            copy = st.selectbox("Return a copy?", options=[True, False], index=0, key='Conversions-truncate-copy')

            if st.checkbox("Apply truncate()", key="Conversions-truncate-apply"):
                try:
                    result = self.data.truncate(before=None if before == "None" else int(before),
                                                after=None if after == "None" else int(after),
                                                axis=axis,
                                                copy=copy)
                    st.write("Resulting DataFrame:", result)

                    # Append to steps, code_snippets, and object_storers
                    st.session_state['steps'].append('truncate')
                    st.session_state['code_snippets'].append(f"df.truncate(before={before}, after={after}, axis={axis}, copy={copy})")
                    st.session_state['object_storers'].append(result)

                    # Update dictionary
                    st.session_state['steps_code_mapping']['truncate'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

                    if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key='Conversions-truncate-radio') == 'Yes':
                        st.session_state['dataset'] = result
                except Exception as e:
                    st.error(f"Error: {e}")

class Indexers:
    def __init__(self):
        self.data = st.session_state['dataset']

    def at(self):
        st.subheader("Perform DataFrame.at[]")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.at[]</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for at[]"):
            row = st.text_input("Enter row label", value="", key="Conversions-at-row")
            col = st.text_input("Enter column label", value="", key="Conversions-at-col")

            if st.checkbox("Apply at[]", key="Conversions-at-apply"):
                try:
                    result = self.data.at[row, col]
                    st.write(f"Value at [{row}, {col}]:", result)

                    # Append to session state lists and dictionary
                    st.session_state['steps'].append('at')
                    st.session_state['code_snippets'].append(f"df.at[{row}, {col}]")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['at'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                except Exception as e:
                    st.error(f"Error: {e}")

    def iat(self):
        st.subheader("Perform DataFrame.iat[]")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.iat[]</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for iat[]"):
            row = st.number_input("Enter row index", min_value=0, value=0, key="Conversions-iat-row")
            col = st.number_input("Enter column index", min_value=0, value=0, key="Conversions-iat-col")

            if st.checkbox("Apply iat[]", key="Conversions-iat-apply"):
                try:
                    result = self.data.iat[row, col]
                    st.write(f"Value at [{row}, {col}]:", result)

                    # Append to session state lists and dictionary
                    st.session_state['steps'].append('iat')
                    st.session_state['code_snippets'].append(f"df.iat[{row}, {col}]")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['iat'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                except Exception as e:
                    st.error(f"Error: {e}")

    def loc(self):
        st.subheader("Perform DataFrame.loc[]")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.loc[]</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for loc[]"):
            row_select_type = st.radio("Select row selection method:", options=['Slice', 'List'], key="Conversions-loc-row-select")
            if row_select_type == 'Slice':
                row_slice_start = st.text_input("Row slice start (e.g., 'a')", key="Conversions-loc-row-slice-start")
                row_slice_end = st.text_input("Row slice end (e.g., 'f')", key="Conversions-loc-row-slice-end")
            else:
                row_list = st.text_input("Row list (e.g., ['a', 'b', 'c'])", key="Conversions-loc-row-list")
            
            col_select_type = st.radio("Select column selection method:", options=['Slice', 'List'], key="Conversions-loc-col-select")
            if col_select_type == 'Slice':
                col_slice_start = st.text_input("Column slice start (e.g., 'col1')", key="Conversions-loc-col-slice-start")
                col_slice_end = st.text_input("Column slice end (e.g., 'col3')", key="Conversions-loc-col-slice-end")
            else:
                col_list = st.text_input("Column list (e.g., ['col1', 'col2'])", key="Conversions-loc-col-list")

            if st.checkbox("Apply loc[]", key="Conversions-loc-apply"):
                try:
                    if row_select_type == 'Slice':
                        rows = slice(row_slice_start, row_slice_end)
                    else:
                        rows = eval(row_list)

                    if col_select_type == 'Slice':
                        cols = slice(col_slice_start, col_slice_end)
                    else:
                        cols = eval(col_list)

                    result = self.data.loc[rows, cols]
                    st.write(f"DataFrame result for loc[{rows}, {cols}]:", result)

                    # Append to session state lists and dictionary
                    st.session_state['steps'].append('loc')
                    st.session_state['code_snippets'].append(f"df.loc[{rows}, {cols}]")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['loc'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                except Exception as e:
                    st.error(f"Error: {e}")

    def iloc(self):
        st.subheader("Perform DataFrame.iloc[]")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.iloc[]</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for iloc[]"):
            row_select_type = st.radio("Select row selection method:", options=['Slice', 'List'], key="Conversions-iloc-row-select")
            if row_select_type == 'Slice':
                row_slice_start = st.number_input("Row slice start", min_value=0, value=0, key="Conversions-iloc-row-slice-start")
                row_slice_end = st.number_input("Row slice end", min_value=0, value=0, key="Conversions-iloc-row-slice-end")
            else:
                row_list = st.text_input("Row list (e.g., [0, 1, 2])", key="Conversions-iloc-row-list")
            
            col_select_type = st.radio("Select column selection method:", options=['Slice', 'List'], key="Conversions-iloc-col-select")
            if col_select_type == 'Slice':
                col_slice_start = st.number_input("Column slice start", min_value=0, value=0, key="Conversions-iloc-col-slice-start")
                col_slice_end = st.number_input("Column slice end", min_value=0, value=0, key="Conversions-iloc-col-slice-end")
            else:
                col_list = st.text_input("Column list (e.g., [0, 1, 2])", key="Conversions-iloc-col-list")

            if st.checkbox("Apply iloc[]", key="Conversions-iloc-apply"):
                try:
                    if row_select_type == 'Slice':
                        rows = slice(row_slice_start, row_slice_end)
                    else:
                        rows = eval(row_list)

                    if col_select_type == 'Slice':
                        cols = slice(col_slice_start, col_slice_end)
                    else:
                        cols = eval(col_list)

                    result = self.data.iloc[rows, cols]
                    st.write(f"DataFrame result for iloc[{rows}, {cols}]:", result)

                    # Append to session state lists and dictionary
                    st.session_state['steps'].append('iloc')
                    st.session_state['code_snippets'].append(f"df.iloc[{rows}, {cols}]")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['iloc'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                except Exception as e:
                    st.error(f"Error: {e}")

class Iterators:
    def __init__(self):
        self.data = st.session_state['dataset']

    def _take_data(self, portion='all'):
        """Helper method to handle DataFrame.take() selection."""
        if portion == 'all':
            return self.data
        else:
            with st.expander("Select Data Portion"):
                axis = st.radio("Select axis for DataFrame.take()", options=[0, 1], index=0, key=f"Conversions-{portion}-axis")
                indices = st.text_area("Enter indices for DataFrame.take() (e.g., 0, 1, 2)", key=f"Conversions-{portion}-indices")
                indices_list = [int(x) for x in indices.split(",")]

            try:
                result = self.data.take(indices_list, axis=axis)
                return result
            except Exception as e:
                st.error(f"Error selecting data portion: {e}")
                return None

    def items(self):
        st.subheader("Perform DataFrame.items()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.items()</h4>", unsafe_allow_html=True)

        with st.expander("Items Parameters"):
            conversion_type = st.radio("Convert entire dataset or selected portion?", options=["Entire Dataset", "Selected Portion"], key="Conversions-items-conversion")

        if conversion_type == "Entire Dataset":
            result_data = self.data
        else:
            result_data = self._take_data(portion='items')

        if result_data is not None:
            with st.expander("Conversion Options"):
                conversion_option = st.radio("Convert to", options=["NumPy Array", "List"], key="Conversions-items-convert")
            
            if conversion_option == "NumPy Array":
                result = result_data.to_numpy()
                st.write(result)
            else:
                result = result_data.values.tolist()
                st.write(result)

            if st.checkbox("Apply items()", key="Conversions-items-apply"):
                st.session_state['steps'].append('items')
                st.session_state['code_snippets'].append(f"df.items()")
                st.session_state['object_storers'].append(result)

                st.session_state['steps_code_mapping']['items'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def iterrows(self):
        st.subheader("Perform DataFrame.iterrows()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.iterrows()</h4>", unsafe_allow_html=True)

        with st.expander("Iterrows Parameters"):
            conversion_type = st.radio("Convert entire dataset or selected portion?", options=["Entire Dataset", "Selected Portion"], key="Conversions-iterrows-conversion")

        if conversion_type == "Entire Dataset":
            result_data = self.data
        else:
            result_data = self._take_data(portion='iterrows')

        if result_data is not None:
            with st.expander("Conversion Options"):
                conversion_option = st.radio("Convert to", options=["NumPy Array", "List"], key="Conversions-iterrows-convert")

            if conversion_option == "NumPy Array":
                result = result_data.to_numpy()
                st.write(result)
            else:
                result = result_data.values.tolist()
                st.write(result)

            if st.checkbox("Apply iterrows()", key="Conversions-iterrows-apply"):
                st.session_state['steps'].append('iterrows')
                st.session_state['code_snippets'].append(f"df.iterrows()")
                st.session_state['object_storers'].append(result)

                st.session_state['steps_code_mapping']['iterrows'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def itertuples(self):
        st.subheader("Perform DataFrame.itertuples()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.itertuples()</h4>", unsafe_allow_html=True)

        with st.expander("Itertuples Parameters"):
            conversion_type = st.radio("Convert entire dataset or selected portion?", options=["Entire Dataset", "Selected Portion"], key="Conversions-itertuples-conversion")

        if conversion_type == "Entire Dataset":
            result_data = self.data
        else:
            result_data = self._take_data(portion='itertuples')

        if result_data is not None:
            with st.expander("Conversion Options"):
                conversion_option = st.radio("Convert to", options=["NumPy Array", "List"], key="Conversions-itertuples-convert")

            if conversion_option == "NumPy Array":
                result = result_data.to_numpy()
                st.write(result)
            else:
                result = result_data.values.tolist()
                st.write(result)

            if st.checkbox("Apply itertuples()", key="Conversions-itertuples-apply"):
                st.session_state['steps'].append('itertuples')
                st.session_state['code_snippets'].append(f"df.itertuples()")
                st.session_state['object_storers'].append(result)

                st.session_state['steps_code_mapping']['itertuples'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
class ShapeManipulaters:
    def __init__(self):
        self.data = st.session_state['dataset']

    def keys(self):
        st.subheader("Perform DataFrame.keys()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.keys()</h4>", unsafe_allow_html=True)

        if st.checkbox("Apply keys()", key="Conversions-keys-apply"):
            result = self.data.keys()
            st.write(result)

            st.session_state['steps'].append('keys')
            st.session_state['code_snippets'].append("df.keys()")
            st.session_state['object_storers'].append(result)

            st.session_state['steps_code_mapping']['keys'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def pop(self):
        st.subheader("Perform DataFrame.pop()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.pop()</h4>", unsafe_allow_html=True)

        with st.expander("Pop Parameters"):
            item = st.selectbox("Select column to pop", self.data.columns, key="Conversions-pop-item")

            if st.checkbox("Apply pop()", key="Conversions-pop-apply"):
                result = self.data.pop(item)
                st.write(f"Column popped: {item}")
                st.write("Updated DataFrame:", self.data)

                st.session_state['steps'].append('pop')
                st.session_state['code_snippets'].append(f"df.pop('{item}')")
                st.session_state['object_storers'].append(self.data)

                st.session_state['steps_code_mapping']['pop'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def insert(self):
        st.subheader("Perform DataFrame.insert()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.insert()</h4>", unsafe_allow_html=True)

        with st.expander("Insert Parameters"):
            loc = st.number_input("Location to insert", min_value=0, max_value=len(self.data.columns), value=0, step=1, key="Conversions-insert-loc")
            column_name = st.text_input("New column name", key="Conversions-insert-column")
            value = st.text_area("Enter values for new column (comma-separated or list)", key="Conversions-insert-value")
            allow_duplicates = st.checkbox("Allow duplicate column names?", key="Conversions-insert-allow_duplicates")

            if st.checkbox("Apply insert()", key="Conversions-insert-apply"):
                value_list = eval(value) if value else []
                try:
                    self.data.insert(loc=loc, column=column_name, value=value_list, allow_duplicates=allow_duplicates)
                    st.write(f"Column '{column_name}' inserted at position {loc}.")
                    st.write("Updated DataFrame:", self.data)

                    st.session_state['steps'].append('insert')
                    st.session_state['code_snippets'].append(f"df.insert({loc}, '{column_name}', {value_list}, allow_duplicates={allow_duplicates})")
                    st.session_state['object_storers'].append(self.data)

                    st.session_state['steps_code_mapping']['insert'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                except Exception as e:
                    st.error(f"Error during insert operation: {e}")

class Evaluators:
    def __init__(self):
        self.data = st.session_state['dataset']

    def isin(self):
        st.subheader("Perform DataFrame.isin()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.isin()</h4>", unsafe_allow_html=True)

        with st.expander("isin Parameters"):
            isin_values = st.text_area("Enter values (list, Series, DataFrame, or dict)", key="Conversions-isin-values")
            values = eval(isin_values) if isin_values else None

            if st.checkbox("Apply isin()", key="Conversions-isin-apply"):
                if values is not None:
                    result = self.data.isin(values)
                    st.write("Resulting DataFrame of booleans:", result)

                    st.session_state['steps'].append('isin')
                    st.session_state['code_snippets'].append(f"df.isin({isin_values})")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['isin'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def where(self):
        st.subheader("Perform DataFrame.where()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.where()</h4>", unsafe_allow_html=True)

        with st.expander("Where Parameters"):
            condition = st.text_area("Enter condition (boolean array/DataFrame or callable)", key="Conversions-where-cond")
            other = st.text_area("Enter replacement value (optional)", key="Conversions-where-other", value="None")
            other_value = eval(other) if other != "None" else None

            if st.checkbox("Apply where()", key="Conversions-where-apply"):
                condition_eval = eval(condition) if condition else None
                if condition_eval is not None:
                    result = self.data.where(condition_eval, other=other_value)
                    st.write("Resulting DataFrame:", result)

                    st.session_state['steps'].append('where')
                    st.session_state['code_snippets'].append(f"df.where({condition}, {other})")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['where'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def mask(self):
        st.subheader("Perform DataFrame.mask()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.mask()</h4>", unsafe_allow_html=True)

        with st.expander("Mask Parameters"):
            condition = st.text_area("Enter condition (boolean array/DataFrame or callable)", key="Conversions-mask-cond")
            other = st.text_area("Enter replacement value (optional)", key="Conversions-mask-other", value="None")
            other_value = eval(other) if other != "None" else None

            if st.checkbox("Apply mask()", key="Conversions-mask-apply"):
                condition_eval = eval(condition) if condition else None
                if condition_eval is not None:
                    result = self.data.mask(condition_eval, other=other_value)
                    st.write("Resulting DataFrame:", result)

                    st.session_state['steps'].append('mask')
                    st.session_state['code_snippets'].append(f"df.mask({condition}, {other})")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping']['mask'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

    def query(self):
        st.subheader("Perform DataFrame.query()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.query()</h4>", unsafe_allow_html=True)

        with st.expander("Query Parameters"):
            expr = st.text_area("Enter the query expression - your data frame lies in self.data", key="Conversions-query-expr")

            if st.checkbox("Apply query()", key="Conversions-query-apply"):
                if expr:
                    try:
                        result = self.data.query(eval(expr))
                        st.write(f"DataFrame result for query '{expr}':", result)

                        st.session_state['steps'].append('query')
                        st.session_state['code_snippets'].append(f"df.query('{expr}')")
                        st.session_state['object_storers'].append(result)

                        st.session_state['steps_code_mapping']['query'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]
                    except Exception as e:
                        st.error(f"Error in query execution: {e}")


class BineryOperators:
    def __init__(self):
        self.data = st.session_state['dataset']

    def binary_operation(self, operation):
        """Perform binary operations like add, sub, mul, etc. based on user selection."""
        st.subheader(f"Perform DataFrame.{operation}()")
        st.markdown(f"<h4 style='color: blue;'>You are going to perform DataFrame.{operation}()</h4>", unsafe_allow_html=True)

        # Select only numeric columns for operations
        numeric_data = self.data.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])

        with st.expander(f"{operation.capitalize()} Parameters"):
            # Radio button for selecting input method for 'other'
            input_method = st.radio("Select Input Method for 'other':", 
                                     options=['Upload Dataset', 'Enter Single Value', 'Enter List'], 
                                     key=f"BineryOperators-{operation}-input-method")
            
            other_eval = None

            if input_method == 'Upload Dataset':
                uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"], key=f"BineryOperators-{operation}-upload")
                if uploaded_file is not None:
                    other_df = pd.read_csv(uploaded_file)
                    other_eval = other_df.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).values  # Convert to numpy array for operations
            elif input_method == 'Enter Single Value':
                other = st.text_input("Enter a single numeric value", key=f"BineryOperators-{operation}-single-value")
                try:
                    other_eval = float(other) if other else None  # Convert to float
                except ValueError:
                    st.error("Please enter a valid numeric value.")
            elif input_method == 'Enter List':
                other = st.text_area("Enter a list of values (comma-separated)", key=f"BineryOperators-{operation}-list")
                if other:
                    try:
                        other_eval = [float(x.strip()) for x in other.split(",")]  # Convert to list of floats
                    except ValueError:
                        st.error("Please enter valid numeric values separated by commas.")

            axis = st.selectbox("Select axis (0 for index, 1 for columns)", [0, 1], index=1, key=f"BineryOperators-{operation}-axis")
            level = st.text_area("Enter level (optional)", key=f"BineryOperators-{operation}-level", value="None")
            level_value = eval(level) if level != "None" else None
            fill_value = st.number_input("Enter fill_value (optional)", key=f"BineryOperators-{operation}-fill-value")
            fill_value_value = fill_value if fill_value else None

            if st.checkbox(f"Apply {operation}()", key=f"BineryOperators-{operation}-apply"):
                try:
                    # Perform operation only on numeric data
                    result = getattr(numeric_data, operation)(other=other_eval, axis=axis, level=level_value, fill_value=fill_value_value)
                    st.write(f"Resulting DataFrame after {operation} on numeric columns:", result)

                    # Store results in session state
                    st.session_state['steps'].append(operation)
                    st.session_state['code_snippets'].append(f"df.{operation}({other_eval}, axis={axis}, level={level_value}, fill_value={fill_value_value})")
                    st.session_state['object_storers'].append(result)

                    st.session_state['steps_code_mapping'][operation] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

                except Exception as e:
                    st.error(f"Error applying {operation}: {e}")

    def add(self):
        self.binary_operation('add')

    def sub(self):
        self.binary_operation('sub')

    def mul(self):
        self.binary_operation('mul')

    def div(self):
        self.binary_operation('div')

    def floordiv(self):
        self.binary_operation('floordiv')

    def mod(self):
        self.binary_operation('mod')

    def pow(self):
        self.binary_operation('pow')

    def dot(self):
        """Perform matrix multiplication with DataFrame.dot()"""
        st.subheader("Perform DataFrame.dot()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.dot()</h4>", unsafe_allow_html=True)

        # Select only numeric columns for operations
        numeric_data = self.data.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])

        with st.expander("Dot Parameters"):
            other = st.text_area("Enter 'other' value (DataFrame or Series)", key="BineryOperators-dot-other")

        if st.checkbox("Apply dot()", key="BineryOperators-dot-apply"):
            other_eval = eval(other) if other else None
            try:
                result = numeric_data.dot(other_eval)
                st.write("Resulting DataFrame after dot multiplication on numeric columns:", result)

                # Store results in session state
                st.session_state['steps'].append('dot')
                st.session_state['code_snippets'].append(f"df.dot({other})")
                st.session_state['object_storers'].append(result)

                st.session_state['steps_code_mapping']['dot'] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

            except Exception as e:
                st.error(f"Error applying dot: {e}")


def Dataset(uploaded_file):
    if uploaded_file is not None:
        # Detect file encoding using chardet
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Read the CSV file with the detected encoding
        uploaded_file.seek(0)  # Reset file pointer after reading
        df = pd.read_csv(uploaded_file, encoding=encoding)
        
        # Standardize the datatypes using klib
        df_standardized = klib.convert_datatypes(df)
        
        # Store the result in session state variable 'dataset'
        st.session_state['dataset'] = df_standardized
        
        return df_standardized
    else:
        return None

class CompareThings:
    def __init__(self):
        self.data = st.session_state.get('dataset', pd.DataFrame())

    def compare_operation(self, operation):
        """Perform comparison operations like lt, gt, le, ge, ne, eq based on user selection."""
        st.subheader(f"Perform DataFrame.{operation}()")
        st.markdown(f"<h4 style='color: blue;'>You are going to perform DataFrame.{operation}()</h4>", unsafe_allow_html=True)

        with st.expander(f"{operation.capitalize()} Parameters"):
            # Radio button for selecting input method for 'other'
            input_method = st.radio("Select Input Method for 'other':", 
                                     options=['Upload Dataset', 'Enter Single Value', 'Enter List', 'Enter String Value'], 
                                     key=f"CompareThings-{operation}-input-method")
            
            other_eval = None

            if input_method == 'Upload Dataset':
                uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"], key=f"CompareThings-{operation}-upload")
                if uploaded_file is not None:
                    other_eval = Dataset(uploaded_file)  # Use the Dataset function
            elif input_method == 'Enter Single Value':
                other = st.text_input("Enter a single numeric value", key=f"CompareThings-{operation}-single-value")
                try:
                    other_eval = float(other) if other else None  # Convert to float
                except ValueError:
                    st.error("Please enter a valid numeric value.")
            elif input_method == 'Enter List':
                other = st.text_area("Enter a list of values (comma-separated)", key=f"CompareThings-{operation}-list")
                if other:
                    try:
                        other_eval = [float(x.strip()) for x in other.split(",")]  # Convert to list of floats
                    except ValueError:
                        st.error("Please enter valid numeric values separated by commas.")
            elif input_method == 'Enter String Value':
                other_eval = st.text_input("Enter a string value for comparison", key=f"CompareThings-{operation}-string-value")

            axis = st.selectbox("Select axis (0 for index, 1 for columns)", [0, 1], index=1, key=f"CompareThings-{operation}-axis")
            level = st.text_area("Enter level (optional)", key=f"CompareThings-{operation}-level", value="None")
            level_value = eval(level) if level != "None" else None

        if st.checkbox(f"Apply {operation}()", key=f"CompareThings-{operation}-apply"):
            try:
                # Perform the comparison operation on all columns
                result = getattr(self.data, operation)(other=other_eval, axis=axis, level=level_value)
                st.write(f"Resulting DataFrame after {operation} on all columns:", result)

                # Store results in session state
                st.session_state['steps'].append(operation)
                st.session_state['code_snippets'].append(f"df.{operation}({other_eval}, axis={axis}, level={level_value})")
                st.session_state['object_storers'].append(result)

                st.session_state['steps_code_mapping'][operation] = [st.session_state['code_snippets'][-1], st.session_state['object_storers'][-1]]

            except Exception as e:
                st.error(f"Error applying {operation}: {e}")

    def lt(self):
        self.compare_operation('lt')

    def gt(self):
        self.compare_operation('gt')

    def le(self):
        self.compare_operation('le')

    def ge(self):
        self.compare_operation('ge')

    def ne(self):
        self.compare_operation('ne')

    def eq(self):
        self.compare_operation('eq')
class DataFrameOperations:
    def __init__(self):
        self.data = st.session_state.get('dataset', pd.DataFrame())

    def apply_operation(self, operation):
        """Perform DataFrame operations based on user selection."""
        st.subheader(f"Perform DataFrame.{operation}()")
        
        # Select whether to apply on full dataset or a portion
        apply_option = st.radio("Apply on:", ["Full Dataset", "Portion of Dataset"])

        if apply_option == "Portion of Dataset":
            # Select rows
            row_indices = st.multiselect("Select Rows by Index", options=self.data.index.tolist(), default=self.data.index.tolist())
            # Select columns
            selected_columns = st.multiselect("Select Columns", options=self.data.columns.tolist(), default=self.data.columns.tolist())
            
            # Filter dataset based on selections
            data_to_use = self.data.loc[row_indices, selected_columns]
        else:
            data_to_use = self.data

        # Define parameters based on operation
        with st.expander(f"{operation.capitalize()} Parameters"):
            if operation == "corr":
                method = st.selectbox("Correlation Method", ["pearson", "kendall", "spearman"], index=0)
                min_periods = st.number_input("Minimum Periods", min_value=1, value=1)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "corrwith":
                other = st.text_input("Enter other DataFrame or Series for comparison")
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                drop = st.checkbox("Drop Non-Matching Columns", value=False)
                method = st.selectbox("Correlation Method", ["pearson", "kendall", "spearman"], index=0)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "count":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "cov":
                min_periods = st.number_input("Minimum Periods", value=1)
                ddof = st.number_input("Delta Degrees of Freedom", value=1)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation in ["cummax", "cummin", "cumsum", "cumprod"]:
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
            elif operation == "diff":
                periods = st.number_input("Periods", min_value=1, value=1)
                axis = st.selectbox("Select Axis", [0, 1], index=0)
            elif operation == "eval":
                expr = st.text_input("Expression to Evaluate")
                inplace = st.checkbox("Modify in Place", value=False)
            elif operation == "kurt":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "max":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "mean":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "median":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "mode":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                numeric_only = st.checkbox("Numeric Only", value=False)
                dropna = st.checkbox("Drop NaN Values", value=True)
            elif operation == "pct_change":
                periods = st.number_input("Periods", value=1)
                fill_method = st.selectbox("Fill Method", ["pad", "backfill", "None"], index=0)
                limit = st.number_input("Limit", value=1)
                freq = st.text_input("Frequency (optional)")
            elif operation == "prod":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
                min_count = st.number_input("Min Count", value=0)
            elif operation == "quantile":
                q = st.number_input("Quantile (0-1)", min_value=0.0, max_value=1.0, value=0.5)
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                numeric_only = st.checkbox("Numeric Only", value=False)
                interpolation = st.selectbox("Interpolation Method", ["linear", "lower", "higher", "nearest", "midpoint", "polynomial"], index=0)
            elif operation == "rank":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                method = st.selectbox("Ranking Method", ["average", "min", "max", "first", "dense"], index=0)
                numeric_only = st.checkbox("Numeric Only", value=False)
                na_option = st.selectbox("NA Option", ["keep", "raise", "drop"], index=0)
                ascending = st.checkbox("Ascending", value=True)
                pct = st.checkbox("Percentile Rank", value=False)
            elif operation == "round":
                decimals = st.number_input("Decimal Places", min_value=0, value=0)
            elif operation == "sem":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                ddof = st.number_input("Delta Degrees of Freedom", value=1)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "skew":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "sum":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                numeric_only = st.checkbox("Numeric Only", value=False)
                min_count = st.number_input("Min Count", value=0)
            elif operation == "std":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                ddof = st.number_input("Delta Degrees of Freedom", value=1)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "var":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                skipna = st.checkbox("Skip NaN Values", value=True)
                ddof = st.number_input("Delta Degrees of Freedom", value=1)
                numeric_only = st.checkbox("Numeric Only", value=False)
            elif operation == "nunique":
                axis = st.selectbox("Select Axis", [0, 1], index=0)
                dropna = st.checkbox("Drop NaN Values", value=True)

        if st.checkbox(f"Apply {operation}()", key=f"DataFrameOperations-{operation}-apply"):
            try:
                # Perform the specified operation with relevant parameters
                if operation == "corr":
                    result = data_to_use.corr(method=method, min_periods=min_periods, numeric_only=numeric_only)
                elif operation == "corrwith":
                    # Assuming 'other' is a DataFrame or Series, you might need to handle it accordingly
                    other = eval(other)  # Convert string to actual object (ensure safety)
                    result = data_to_use.corrwith(other, axis=axis, drop=drop, method=method, numeric_only=numeric_only)
                elif operation == "count":
                    result = data_to_use.count(axis=axis, numeric_only=numeric_only)
                elif operation == "cov":
                    result = data_to_use.cov(min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)
                elif operation == "cummax":
                    result = data_to_use.cummax(axis=axis, skipna=skipna)
                elif operation == "cummin":
                    result = data_to_use.cummin(axis=axis, skipna=skipna)
                elif operation == "cumsum":
                    result = data_to_use.cumsum(axis=axis, skipna=skipna)
                elif operation == "cumprod":
                    result = data_to_use.cumprod(axis=axis, skipna=skipna)
                elif operation == "diff":
                    result = data_to_use.diff(periods=periods, axis=axis)
                elif operation == "eval":
                    result = data_to_use.eval(expr, inplace=inplace)
                elif operation == "kurt":
                    result = data_to_use.kurt(axis=axis, skipna=skipna, numeric_only=numeric_only)
                elif operation == "max":
                    result = data_to_use.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
                elif operation == "mean":
                    result = data_to_use.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
                elif operation == "median":
                    result = data_to_use.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
                elif operation == "mode":
                    result = data_to_use.mode(axis=axis, numeric_only=numeric_only, dropna=dropna)
                elif operation == "pct_change":
                    result = data_to_use.pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq)
                elif operation == "prod":
                    result = data_to_use.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
                elif operation == "quantile":
                    result = data_to_use.quantile(q=q, axis=axis, numeric_only=numeric_only, interpolation=interpolation)
                elif operation == "rank":
                    result = data_to_use.rank(axis=axis, method=method, numeric_only=numeric_only, na_option=na_option, ascending=ascending, pct=pct)
                elif operation == "round":
                    result = data_to_use.round(decimals=decimals)
                elif operation == "sem":
                    result = data_to_use.sem(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only)
                elif operation == "skew":
                    result = data_to_use.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
                elif operation == "sum":
                    result = data_to_use.sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
                elif operation == "std":
                    result = data_to_use.std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only)
                elif operation == "var":
                    result = data_to_use.var(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only)
                elif operation == "nunique":
                    result = data_to_use.nunique(axis=axis, dropna=dropna)
                else:
                    st.error(f"Unsupported operation: {operation}")
                    return

                st.write(f"Result of DataFrame.{operation}():", result)

                # Store results in session state
                st.session_state['steps'].append(operation)
                st.session_state['code_snippets'].append(f"df.{operation}()")
                st.session_state['object_storers'].append(result)

            except Exception as e:
                st.error(f"Error applying {operation}: {e}")

    def execute(self):
        """Execute DataFrame operations based on user input."""
        operation = st.selectbox("Select DataFrame operation", 
                                  ["corr", "corrwith", "count", "cov", "cummax", "cummin", 
                                   "cumsum", "cumprod", "diff", "eval", "kurt", "max", 
                                   "mean", "median", "mode", "pct_change", "prod", "quantile", 
                                   "rank", "round", "sem", "skew", "sum", "std", "var", "nunique"])
        self.apply_operation(operation)

class MissingDataHandler:
    def __init__(self):
        self.data = st.session_state['dataset']

    def bfill(self):
        """Backward fill missing values in the DataFrame."""
        st.subheader("Perform DataFrame.bfill()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.bfill()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for bfill()"):
            axis = st.selectbox("Select axis", options=[0, 1], index=0, key='MissingDataHandler-bfill-axis')
            inplace = st.checkbox("In-place operation?", value=False, key='MissingDataHandler-bfill-inplace')
            limit = st.number_input("Limit for filling (None for no limit)", value=None, key='MissingDataHandler-bfill-limit')

            if st.checkbox("Apply bfill()", key="MissingDataHandler-bfill-apply"):
                original_data = self.data.copy()
                result = self.data.bfill(axis=axis, inplace=inplace, limit=limit)
                st.dataframe(result)

                if not inplace:
                    self.data = result

                self.log_result("Backward Fill", original_data, self.data)

    def ffill(self):
        """Forward fill missing values in the DataFrame."""
        st.subheader("Perform DataFrame.ffill()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.ffill()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for ffill()"):
            axis = st.selectbox("Select axis", options=[0, 1], index=0, key='MissingDataHandler-ffill-axis')
            inplace = st.checkbox("In-place operation?", value=False, key='MissingDataHandler-ffill-inplace')
            limit = st.number_input("Limit for filling (None for no limit)", value=None, key='MissingDataHandler-ffill-limit')

            if st.checkbox("Apply ffill()", key="MissingDataHandler-ffill-apply"):
                original_data = self.data.copy()
                result = self.data.ffill(axis=axis, inplace=inplace, limit=limit)
                st.dataframe(result)

                if not inplace:
                    self.data = result

                self.log_result("Forward Fill", original_data, self.data)

    def fillna(self):
        """Fill missing values with a specified value or method."""
        st.subheader("Perform DataFrame.fillna()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.fillna()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for fillna()"):
            value = st.text_area("Enter value to fill (e.g., 0)", key='MissingDataHandler-fillna-value', value='0')
            method = st.selectbox("Fill method", options=[None, 'ffill', 'bfill'], index=0, key='MissingDataHandler-fillna-method')
            axis = st.selectbox("Select axis", options=[0, 1], index=0, key='MissingDataHandler-fillna-axis')
            inplace = st.checkbox("In-place operation?", value=False, key='MissingDataHandler-fillna-inplace')
            limit = st.number_input("Limit for filling (None for no limit)", value=None, key='MissingDataHandler-fillna-limit')

            if st.checkbox("Apply fillna()", key="MissingDataHandler-fillna-apply"):
                original_data = self.data.copy()
                result = self.data.fillna(value=eval(value) if value != '' else None, 
                                           method=method, axis=axis, 
                                           inplace=inplace, limit=limit)
                st.dataframe(result)

                if not inplace:
                    self.data = result
                    st.dataframe(result)

                self.log_result("Fill NaN", original_data, self.data)

    def interpolate(self):
        """Interpolate missing values in the DataFrame."""
        st.subheader("Perform DataFrame.interpolate()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.interpolate()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for interpolate()"):
            method = st.selectbox("Interpolation method", options=['linear', 'time', 'index', 'nearest', 'polynomial', 'spline'], index=0, key='MissingDataHandler-interpolate-method')
            axis = st.selectbox("Select axis", options=[0, 1], index=0, key='MissingDataHandler-interpolate-axis')
            limit = st.number_input("Limit for interpolation (None for no limit)", value=None, key='MissingDataHandler-interpolate-limit')
            limit_direction = st.selectbox("Limit direction", options=[None, 'forward', 'backward'], index=0, key='MissingDataHandler-interpolate-limit_direction')

            if st.checkbox("Apply interpolate()", key="MissingDataHandler-interpolate-apply"):
                original_data = self.data.select_dtypes(include=["int32","int64","float32","float64"]).copy()
                result = self.data.select_dtypes(include=["int32","int64","float32","float64"]).interpolate(method=method, axis=axis, limit=limit, 
                                               limit_direction=limit_direction)
                st.dataframe(result)

                self.data = result
                self.log_result("Interpolate", original_data, self.data)

    def isna(self):
        """Check for missing values in the DataFrame."""
        st.subheader("Check for Missing Values")
        missing_mask = self.data.isna()
        st.write("Missing Values Mask:")
        st.dataframe(missing_mask)
        

    def notnull(self):
        """Check for non-missing values in the DataFrame."""
        st.subheader("Check for Non-Missing Values")
        non_missing_mask = self.data.notnull()
        st.write("Non-Missing Values Mask:")
        st.dataframe(non_missing_mask)
        self.log_result("Not Null", self.data,non_missing_mask)

    def replace(self):
        """Replace values in the DataFrame."""
        st.subheader("Perform DataFrame.replace()")
        st.markdown("<h4 style='color: blue;'>You are going to perform DataFrame.replace()</h4>", unsafe_allow_html=True)

        with st.expander("Customize Parameters for replace()"):
            to_replace = st.text_area("Enter values to replace (e.g., {'old_value': 'new_value'})", key='MissingDataHandler-replace-to_replace', value="{'old_value': 'new_value'}")
            value = st.text_area("Enter new value (or leave empty to keep old)", key='MissingDataHandler-replace-value', value='')

            inplace = st.checkbox("In-place operation?", value=False, key='MissingDataHandler-replace-inplace')
            limit = st.number_input("Limit for replacements (None for no limit)", value=None, key='MissingDataHandler-replace-limit')
            regex = st.checkbox("Use regex for replacements?", value=False, key='MissingDataHandler-replace-regex')

            if st.checkbox("Apply replace()", key="MissingDataHandler-replace-apply"):
                original_data = self.data.copy()
                result = self.data.replace(to_replace=eval(to_replace) if to_replace != '' else None, 
                                           value=eval(value) if value != '' else None, 
                                           inplace=inplace, limit=limit, regex=regex)
                st.dataframe(result)

                if not inplace:
                    self.data = result

                self.log_result("Replace", original_data, self.data)

    def log_result(self, operation_name, original_data, updated_data):
        """Log the operation result to the session state."""
        # Store the operation details in session state
        step_description = f"{operation_name} was applied."
        code_snippet = f"{operation_name}()"
        object_storer = updated_data

        st.session_state.steps.append(step_description)
        st.session_state.code_snippets.append(code_snippet)
        st.session_state.object_storers.append(object_storer)

        if st.radio("Set this as the session dataset?", options=['Yes', 'No'], index=1, key=f'MissingDataHandler-{operation_name}-radio') == 'Yes':
            st.session_state['dataset'] = updated_data

# FUNCTION DEFINITION GOES HERE
def Dataset(uploaded_file):
    if uploaded_file is not None:
        # Detect file encoding using chardet
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Read the CSV file with the detected encoding
        uploaded_file.seek(0)  # Reset file pointer after reading
        df = pd.read_csv(uploaded_file, encoding=encoding)
        
        # Standardize the datatypes using klib
        df_standardized = klib.convert_datatypes(df)
        
        # Store the result in session state variable 'dataset'
        st.session_state['dataset'] = df_standardized
        
        return df_standardized
    else:
        return None
    
upload_data = st.sidebar.file_uploader("Upload Any CSV File Only", type=["csv"])
if upload_data:
    st.session_state['dataset'] = Dataset(upload_data)
    st.write("Dataset loaded successfully!")

    with st.sidebar:
        selected = option_menu(
            menu_title=None, 
            options=["Perform Operations"],  
            icons=["bar-chart", "clock-history", "terminal"],  
            menu_icon="cast",  
            default_index=0 
        )

    if selected == "Perform Operations":
        # Instantiate Conversions class
        conversions = Conversions()

        # Conversions operations
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Conversions</h3>", unsafe_allow_html=True)

        # Checkboxes for each operation
        astype_checkbox = st.sidebar.checkbox('DataFrame.astype()')
        if astype_checkbox:
            conversions.astype()  # Use the instance to call the method

        convert_dtypes_checkbox = st.sidebar.checkbox('DataFrame.convert_dtypes()')
        if convert_dtypes_checkbox:
            conversions.convert_dtypes()  # Use the instance to call the method

        infer_objects_checkbox = st.sidebar.checkbox('DataFrame.infer_objects()')
        if infer_objects_checkbox:
            conversions.infer_objects()  # Use the instance to call the method

        copy_checkbox = st.sidebar.checkbox('DataFrame.copy()')
        if copy_checkbox:
            conversions.copy()  # Use the instance to call the method

        to_numpy_checkbox = st.sidebar.checkbox('DataFrame.to_numpy()')
        if to_numpy_checkbox:
            conversions.to_numpy()  # Use the instance to call the method

        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>First N Rows And N Columns</h3>", unsafe_allow_html=True)
        head=st.sidebar.checkbox("DataFrame.head()")
        if head:
            FirstNRows().head()
        tail=st.sidebar.checkbox("DataFrame.tail()")
        if tail:
            FirstNRows().tail()
        take=st.sidebar.checkbox("DataFrame.take()")
        if take:
            FirstNRows().take()
        truncate=st.sidebar.checkbox("DataFrame.truncate()")
        if truncate:
            FirstNRows().truncate()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Indexers</h3>", unsafe_allow_html=True)
        at=st.sidebar.checkbox("DataFrame.at()")
        if at:
            Indexers().at()
        iat=st.sidebar.checkbox("DataFrame.iat()")
        if iat:
            Indexers().iat()
        loc=st.sidebar.checkbox("DataFrame.loc()")
        if loc:
            Indexers().loc()
        iloc=st.sidebar.checkbox("DataFrame.iloc")
        if iloc:
            Indexers().iloc()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Iterators</h3>", unsafe_allow_html=True)
        items=st.sidebar.checkbox("DataFrame.items()")
        if items:
            Iterators().items()
        iterrows=st.sidebar.checkbox("DataFrame.iterrows")
        if iterrows:
            Iterators().iterrows()
        itertuples=st.sidebar.checkbox("DataFrame.itertuples()")
        if itertuples:
            Iterators().itertuples()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Shape Manipulaters</h3>", unsafe_allow_html=True)
        keys=st.sidebar.checkbox("DataFrame.keys()")
        if keys:
           ShapeManipulaters().keys() 
        pop=st.sidebar.checkbox("DataFrame.pop()")
        if pop:
            ShapeManipulaters().pop()
        insert=st.sidebar.checkbox("DataFrame.insert()")
        if insert:
            ShapeManipulaters().insert()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Evaluators</h3>", unsafe_allow_html=True)
        query=st.sidebar.checkbox("Query")
        if query:
            Evaluators().query()
        where=st.sidebar.checkbox("Where")
        if where:
            Evaluators().where()
        isin=st.sidebar.checkbox("IsIn")
        if isin:
            Evaluators().isin()
        mask=st.sidebar.checkbox("Mask")
        if mask:
            Evaluators().mask()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Binery Operators</h3>", unsafe_allow_html=True)
        add=st.sidebar.checkbox("DataFrame.add()")
        if add:
            BineryOperators().add()
        sub=st.sidebar.checkbox("DataFrame.sub()")
        if sub:
            BineryOperators().sub()
        mul=st.sidebar.checkbox("DataFrame.mul()")
        if mul:
            BineryOperators().mul()
        div=st.sidebar.checkbox("DataFrame.div()")
        if div:
            BineryOperators().div()
        floordiv=st.sidebar.checkbox("DataFrame.floordiv()")
        if floordiv:
            BineryOperators().floordiv()
        mod=st.sidebar.checkbox("DataFrame.mod()")
        if mod:
            BineryOperators().mod()
        Pow=st.sidebar.checkbox("DataFrame.pow()")
        if Pow:
            BineryOperators().pow()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Element Wise Comparision</h3>", unsafe_allow_html=True)
        eq=st.sidebar.checkbox("DataFrame.eq()")
        if eq:
            CompareThings().eq()
        lt=st.sidebar.checkbox("DataFrame.lt()")
        if lt:
            CompareThings().lt()
        gt=st.sidebar.checkbox("DataFrame.gt()")
        if gt:
            CompareThings().gt()
        ne=st.sidebar.checkbox("DataFrame.ne()")
        if ne:
            CompareThings().ne()
        le=st.sidebar.checkbox("DataFrame.le()")
        if le:
            CompareThings().le()
        ge=st.sidebar.checkbox("DataFrame.ge()")
        if ge:
            CompareThings().ge()
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Descriptive Statistics</h3>", unsafe_allow_html=True)
        operations = [
            "corr", "corrwith", "count", "cov", "cummax", "cummin",
            "cumsum", "cumprod", "diff", "eval", "kurt", "max",
            "mean", "median", "mode", "pct_change", "prod", "quantile",
            "rank", "round", "sem", "skew", "sum", "std", "var", "nunique"
        ]
        df_ops = DataFrameOperations()
        for operation in operations:
            if st.sidebar.checkbox(operation):
                df_ops.apply_operation(operation)
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: yellow;'>Missing Data Handler</h3>", unsafe_allow_html=True)
        bfill=st.sidebar.checkbox("DataFrame.bfill()")
        if bfill:
            MissingDataHandler().bfill()
        ffill=st.sidebar.checkbox("DataFrame.ffill()")
        if ffill:
            MissingDataHandler().ffill()
        fillna=st.sidebar.checkbox("DataFrame.fillna()")
        if fillna:
            MissingDataHandler().fillna()
        interpolate=st.sidebar.checkbox("DataFrame.interpolate()")
        if interpolate:
            MissingDataHandler().interpolate()
        isna=st.sidebar.checkbox("DataFrame.isna()")
        if isna:
            MissingDataHandler().isna()
        notnull=st.sidebar.checkbox("DataFrame.notnull()")
        if notnull:
            MissingDataHandler().notnull()
        replace=st.sidebar.checkbox("DataFrame.replace()")
        if replace:
            MissingDataHandler().replace()
        
