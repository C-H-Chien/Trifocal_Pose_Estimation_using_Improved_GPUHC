%> Transform parametric polynomials into coefficient polynomials
%> An Improved Version with Minimal Number of Coefficients by Removing
%  Duplicate Coefficients With Different Signs.
%
%> Chiang-Heng Chien
%> Initiated                        Oct. 19th, 2022
%> Solve Issues                     Nov. 6th,  2022
%> Solve a Factorization Issue      Nov. 11th, 2022
%> Solve a Parameter Homotopy Issue Nov. 20th, 2022

clear;
format long;

%> This is my lab laptop directory
%fileFolder = '/path/to/your/minimal-problems/';
fileFolder = '/oscar/data/bkimia/cchien3/Trifocal_Relative_Pose_Estimator/';
problem = "trifocal_2op1p/";
problemName = extractBefore(problem, "/");

params.make_coefficientHC = 0;
params.write_Ht_indices = 1;
params.write_Hx_indices = 1;

if params.write_Ht_indices
    outputHtIndicesFileName = 'Ht_indices.txt';
    fullOutputFileName_Ht_indices = fullfile(fileFolder, problem, outputHtIndicesFileName);
    outputFileWr_Ht_indices = fopen(fullOutputFileName_Ht_indices, 'w');
end

if params.write_Hx_indices
    outputHxIndicesFileName = 'Hx_indices.txt';
    fullOutputFileName_Hx_indices = fullfile(fileFolder, problem, outputHxIndicesFileName);
    outputFileWr_Hx_indices = fopen(fullOutputFileName_Hx_indices, 'w');
end

outputFileName_p2c = 'optimal_params2coeffs.txt';
outputFileName_coefficient_problem = 'optimal_rep_problem.txt';
%outputFileName_P2C_script = 'P2C_script.txt';
outputFileName_P2C_Hx_script = 'P2C_gen_Hx.txt';
outputFileName_P2C_Ht_script = 'P2C_gen_Ht.txt';
fullOutputFileName_p2c = fullfile(fileFolder, problem, outputFileName_p2c);
fullOutputFileName_rep_problem = fullfile(fileFolder, problem, outputFileName_coefficient_problem);
%fullOutputFileName_p2c_script = fullfile(fileFolder, problem, outputFileName_P2C_script);
fullOutputFileName_p2c_Hx_script = fullfile(fileFolder, problem, outputFileName_P2C_Hx_script);
fullOutputFileName_p2c_Ht_script = fullfile(fileFolder, problem, outputFileName_P2C_Ht_script);
outputFileWr_p2c = fopen(fullOutputFileName_p2c, 'w');
outputFileWr_rep_problem = fopen(fullOutputFileName_rep_problem, 'w');
%outputFileWr_P2C_script = fopen(fullOutputFileName_p2c_script, 'w');
outputFileWr_P2C_Hx_script = fopen(fullOutputFileName_p2c_Hx_script, 'w');
outputFileWr_P2C_Ht_script = fopen(fullOutputFileName_p2c_Ht_script, 'w');

%> Optional... write replaced, reordered problem which keeps the parameters
outputFileName_keep_params_rep_problem = 'optimal_rep_problem_keep_params.txt';
fullOutputFileName_keep_params_rep_problem = fullfile(fileFolder, problem, outputFileName_keep_params_rep_problem);
outputFileWr_keep_params_rep_problem = fopen(fullOutputFileName_keep_params_rep_problem, 'w');


% -- define systems --
if strcmp(problem, 'trifocal_2op1p/')
    [f, numOfVars] = sys_trifocal_2op1p_30();
end

for i = 1:numOfVars
    str_vars = 'x';
    cat_str_vars = strcat(str_vars, num2str(i));
    X(i) = str2sym(cat_str_vars);
end

% ========================================================================
% > Find unique coefficients from compositions of parameters
% ========================================================================
%> How to find unique coefficients, regardless of the signs?
%> Stack all possible coefficients, then use unique function to find them
fprintf("Finding unique coefficients from all the polynomial equations ...\n");
coeffs_stack = [];
for p = 1:numOfVars
    %> extract coefficients and variables
    [coefficient, variable] = coeffs(f(p), X);
    
    %> loop over all coefficients
    for ci = 1:size(coefficient, 2)

        %> target coefficient
        %str_coeff = string(coefficient(ci));
        %parts = strsplit(str_coeff, "*");

        %> first factor out for checking the existence of a scalar
        factors = factor(coefficient(ci));
        parts = string(factors);
        
        %[ci, coefficient(ci)]
        %factors
        
        %> check if there is a scalar
        has_s = 0;
        has_p = 0;
        for pi = 1:size(parts, 2)
            if contains(parts(pi), 'p')
                %> if there is a parameter                
                has_p = 1;
                break;
            else
                %> if there is a scalar
                has_s = 1;
                scalar = parts(pi);
            end
        end
        
        %> Check if an expansion is necessary for the factorization result
        %  Typically, "factors" has only two entries, one for scalar and
        %  the other for the parameter combinations; thus checking the size
        %  of the factors enables us to know the expansion requirement
        start_cpi_idx = 1;
        store_coeff = "";
        if has_s && size(factors, 2) > 2
            sym_coeffs = 1;
            for cpi = 2:size(factors, 2)
                sym_coeffs = sym_coeffs * factors(cpi);
            end
            sym_store_coeff = expand(sym_coeffs);
            store_coeff = string(sym_store_coeff);
        else
            if has_p
                if has_s
                    start_cpi_idx = 2;
                end

                for cpi = start_cpi_idx : size(parts, 2)
                    if cpi > start_cpi_idx
                        store_coeff = strcat(store_coeff, "*", parts(cpi));
                    else
                        store_coeff = parts(cpi);
                    end
                end
            end
        end

        %> stack all possible coefficients (in terms of parameters)
        if ~strcmp(store_coeff, "")
            coeffs_stack = [coeffs_stack, store_coeff];
        end
    end
end

%> find the uniqueness of the coefficients
[unique_coeffs, ~] = unique(coeffs_stack);
%uniqueIdx = sort(uniqueIdx, 'ascend');
%uni = coeffs_stack(uniqueIdx);

fprintf("Complete writing individual coefficients to a file ");
rep_coeffs_stack = strings(size(unique_coeffs, 2), 1);
for i = 1:size(unique_coeffs, 2)
    
    if params.make_coefficientHC
        rep_coeffs_stack(i,1) = strcat('h_coeffs[', num2str(i-1), ']');
    else
        rep_coeffs_stack(i,1) = strcat('c(', num2str(i), ')');
    end
    
    rep_coeffs_stack(i,1) = strcat(rep_coeffs_stack(i,1), {' '}, '=', {' '}, unique_coeffs(i));
    
    if params.make_coefficientHC
        rep_coeffs_stack(i,1) = strcat(rep_coeffs_stack(i,1), ';');
    end

    fprintf(outputFileWr_p2c, rep_coeffs_stack(i,1));
    fprintf(outputFileWr_p2c, '\n');
    fprintf('. ');
end
fprintf('\n');

%> create a P2C script which converts parameter homotopy as a uni-variable
%  polynomial for coefficient representation
%createBlankP2C_cpp(problemName, outputFileWr_P2C_script);
max_order_of_func_t = createUnivPoly4Coeffs(rep_coeffs_stack, outputFileWr_P2C_Hx_script, outputFileWr_P2C_Ht_script);
% fprintf(outputFileWr_P2C_script, "}\n");
% fprintf(outputFileWr_P2C_script, "#endif\n");

for i = 1:size(unique_coeffs, 2)+1
    str_coeffs = 'c';
    cat_str_coeffs = strcat(str_coeffs, num2str(i));
    C(i) = str2sym(cat_str_coeffs);
end

% ========================================================================
% > Construct a replaced polynomial formulations
% ========================================================================
%> Generate polynomials with replaced coefficients
fprintf("Constructing replaced polynomials ...\n");
replaced_polys = strings(numOfVars, 1);
replaced_polys_keep_params = strings(numOfVars, 1);
max_num_of_vars_per_term = 0;
max_num_of_terms_per_poly = 0;
max_num_of_terms_per_Hx_poly = 0;
for p = 1:numOfVars
    %> extract coefficients and variables
    [coefficient, variable] = coeffs(f(p), X);
    
    poly_reordered_terms = '';
    poly_reordered_terms_keep_params = '';
    
    %> find the maximal number of terms per polynomial
    if size(coefficient, 2) > max_num_of_terms_per_poly
        max_num_of_terms_per_poly = size(coefficient, 2);
    end
    
    %> loop over all coefficients
    for ci = 1:size(coefficient, 2)
        
        %> target variable
        str_var = string(variable(ci));
        
        %> find the maximal number of variables in each term
        [has_power, expanded_vars] = expand_powers_in_poly_term(str_var);
        indivisual_var = strsplit(expanded_vars, "*");
        if size(indivisual_var, 2) > max_num_of_vars_per_term
            max_num_of_vars_per_term = size(indivisual_var, 2);
        end
        
        %> target coefficient
%         orig_str_coeff = string(coefficient(ci));
%         str_coeff = string(coefficient(ci));
%         parts = strsplit(str_coeff, "*");
        %coefficient(ci)
        
        %> check if it is a pure scalar
        has_s = 0;
        has_p = 0;
        scalar = "";
        if contains(string(coefficient(ci)), 'p')
            factors = factor(coefficient(ci));
            parts = string(factors);
            
            %[ci, coefficient(ci)]
            %factors
            
            for pi = 1:size(parts, 2)
                if contains(parts(pi), 'p')
                    %> if there is a parameter                
                    has_p = 1;
                    break;
                else
                    %> if there is a scalar
                    has_s = 1;
                    scalar = parts(pi);
                end
            end
            
%             %> check if there is a scalar
%             has_s = 0;
%             has_p = 0;
%             scalar = "";
%             for pi = 1:size(parts, 2)
%                 if contains(parts(pi), 'p')
%                     %> if there is a parameter
%                     has_p = 1;
%                     break;
%                 else
%                     %> if there is a scalar
%                     has_s = 1;
%                     scalar = parts(pi);
%                 end
%             end
        else
            has_s = 1;
            factors = "";
            scalar = string(coefficient(ci));
        end
        % -----------------------------------------------------------
       
        %> Check if an expansion is necessary for the factorization result
        %  Typically, "factors" has only two entries, one for scalar and
        %  the other for the parameter combinations; thus checking the size
        %  of the factors enables us to know the expansion requirement
        start_cpi_idx = 1;
        target_coeff = "";
        str_coeff = "";
        if has_s && size(factors, 2) > 2
            sym_coeffs = 1;
            for cpi = 2:size(factors, 2)
                sym_coeffs = sym_coeffs * factors(cpi);
            end
            sym_store_coeff = expand(sym_coeffs);
            target_coeff = string(sym_store_coeff);
            
            str_coeff = strcat(scalar, "*(", target_coeff, ")");
        else
            if has_p
                if has_s
                    start_cpi_idx = 2;
                    str_coeff = strcat(str_coeff, scalar, "*(");
                else
                    str_coeff = strcat(str_coeff, "(");
                end

                for cpi = start_cpi_idx : size(parts, 2)
                    if cpi > start_cpi_idx
                        target_coeff = strcat(target_coeff, "*", parts(cpi));
                    else
                        target_coeff = parts(cpi);
                    end
                end
                
                str_coeff = strcat(str_coeff, target_coeff, ")");
            else
                str_coeff = "()";
            end
        end
        
        % -----------------------------------------------------------
%         start_cpi_idx = 1;
%         target_coeff = "";
%         str_coeff = "";
%         if has_p
%             if has_s
%                 start_cpi_idx = 2;
%                 str_coeff = strcat(str_coeff, scalar, "*(");
%             end
%             
%             for cpi = start_cpi_idx : size(parts, 2)
%                 if cpi > start_cpi_idx
%                     target_coeff = strcat(target_coeff, "*", parts(cpi));
%                 else
%                     target_coeff = parts(cpi);
%                 end
%             end
%             
%             str_coeff = strcat(str_coeff, target_coeff, ")");
%         end
        
        %> Now we have the "target coefficient" for each terms, we can 
        %  replace it in each term with a coefficient
        if ~strcmp(target_coeff, "")
            unique_idx = strmatch(target_coeff, unique_coeffs, "exact");
            replace_str = strcat("c", string(unique_idx));
            replace_coeff = strrep(str_coeff, target_coeff, replace_str);
            replace_coeff = strrep(replace_coeff, "(", "");
            replace_coeff = strrep(replace_coeff, ")", "");
            
            if has_s
                replace_coeff_keep_params = strcat(scalar, "*(", target_coeff, ")*");
            else
                replace_coeff_keep_params = strcat("(", target_coeff, ")*");
            end
            
            if strcmp(str_var, "1")
%                 if contains(orig_str_coeff, "-")
%                     poly_reordered_terms = strcat(poly_reordered_terms, " - ", replace_coeff, "*", str_var);
%                 else
%                     if ci > 1
%                         poly_reordered_terms = strcat(poly_reordered_terms, "+", replace_coeff, "*", str_var);
%                     else
%                         poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, "*", str_var);
%                     end
%                 end
                
                if contains(scalar, "-")
                    poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, "*", str_var);
                    poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, replace_coeff_keep_params, str_var);
                else
                    if ci > 1
                        poly_reordered_terms = strcat(poly_reordered_terms, "+", replace_coeff, "*", str_var);
                        poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, "+", replace_coeff_keep_params, str_var);
                    else
                        poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, "*", str_var);
                        poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, replace_coeff_keep_params, str_var);
                    end
                end
            else
                if contains(scalar, "-")
                    poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, "*", str_var);
                    poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, replace_coeff_keep_params, str_var);
                else
                    if ci > 1
                        poly_reordered_terms = strcat(poly_reordered_terms, "+", replace_coeff, "*", str_var);
                        poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, "+", replace_coeff_keep_params, str_var);
                    else
                        poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, "*", str_var);
                        poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, replace_coeff_keep_params, str_var);
                    end
                end
            end
        else
            if has_s
                replace_coeff = strcat(scalar, "*c", string(size(unique_coeffs,2)+1));
                
                replace_coeff_keep_params = strcat(scalar, "*(", target_coeff, ")*");
            else
                replace_coeff_keep_params = strcat("(", target_coeff, ")*");
%             else
%                 replace_str = strcat("c", string(size(unique_coeffs,2)+1));
%                 replace_coeff = strrep(orig_str_coeff, "1", replace_str);
            end
            
            if contains(scalar, "-")
                poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, '*', str_var);
                poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, replace_coeff_keep_params, str_var);
            else
                if ci > 1
                    poly_reordered_terms = strcat(poly_reordered_terms, "+", replace_coeff, '*', str_var);
                    poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, "+", replace_coeff_keep_params, str_var);
                else
                    poly_reordered_terms = strcat(poly_reordered_terms, replace_coeff, '*', str_var);
                    poly_reordered_terms_keep_params = strcat(poly_reordered_terms_keep_params, replace_coeff_keep_params, str_var);
                end
            end
        end
    end
    
    poly_reordered_terms = strrep(poly_reordered_terms, "+", " + ");
    poly_reordered_terms = strrep(poly_reordered_terms, "-", " - ");    
    
    poly_reordered_terms_keep_params = strrep(poly_reordered_terms_keep_params, "+", " + ");
    poly_reordered_terms_keep_params = strrep(poly_reordered_terms_keep_params, "-", " - ");
    
    %> store in a string array
    replaced_polys(p,1) = poly_reordered_terms;
    
    %> find the maximal number of terms per Jacobian Hx polynomial
    for dv = 1:numOfVars
        J = jacobian(str2sym(poly_reordered_terms), X(dv));
        y = strsplit(string(J), {'+', '-'});
        if size(y,2) > max_num_of_terms_per_Hx_poly
            max_num_of_terms_per_Hx_poly = size(y,2);
        end
    end
    
    replaced_polys_keep_params(p,1) = poly_reordered_terms_keep_params;
    
    fprintf(outputFileWr_rep_problem, poly_reordered_terms);
    fprintf(outputFileWr_rep_problem, '\n');
    
    fprintf(outputFileWr_keep_params_rep_problem, poly_reordered_terms_keep_params);
    fprintf(outputFileWr_keep_params_rep_problem, '\n');
end

max_num_of_parts_per_term = 2 + max_num_of_vars_per_term;
% ========================================================================
% > Extract indices of the Jacobian Ht
% ========================================================================
%> Now we can extract the indices of the Jacobian Ht
fprintf("Extracting indices of the Jacobian Ht ...\n");
Ht_indices = zeros(numOfVars, max_num_of_terms_per_poly*max_num_of_parts_per_term);
for p = 1:numOfVars
    target_poly = str2sym(replaced_polys(p,1));
    
    %> isolate variables
    [coefficients, variables] = coeffs(target_poly, X);
    
    %> loop over all coefficients
    for ci = 1:size(coefficients, 2)
        %> define the starting index
        begin_idx = (ci-1)*max_num_of_parts_per_term+1;
        
        %> for each coefficient, isolate scalars
        [scalars, rep_cs] = coeffs(coefficients(1,ci), C);

        %> push the scalar
        Ht_indices(p, begin_idx) = scalars;
        
        %> push the coefficient index matched with c++
        c_idx = extractAfter(string(rep_cs), "c");
        cpp_c_idx = double(c_idx)-1;
        Ht_indices(p, begin_idx+1) = cpp_c_idx;
        
        %> for the variable, check the power first
        str_var = string(variables(1,ci));
        [~, expanded_vars] = expand_powers_in_poly_term(str_var);
        vars_parts = strsplit(expanded_vars, "*");
        
        %> push the variable indices matched with c++
        if strcmp(vars_parts, "1")
            remain_num_of_vars = max_num_of_vars_per_term;
            for ri = 1:remain_num_of_vars
                Ht_indices(p, begin_idx+size(vars_parts, 2)+ri) = numOfVars;
            end
            continue;
        else
            for vi = 1:size(vars_parts, 2)
                v_idx = extractAfter(vars_parts(1,vi), "x");
                cpp_v_idx = double(v_idx)-1;
                Ht_indices(p, begin_idx+1+vi) = cpp_v_idx;
            end

            if size(vars_parts, 2) < max_num_of_vars_per_term
                remain_num_of_vars = max_num_of_vars_per_term - size(vars_parts, 2);
                for ri = 1:remain_num_of_vars
                    Ht_indices(p, begin_idx+size(vars_parts, 2)+1+ri) = numOfVars;
                end
            end
        end        
    end
    
    %> append the rest of the terms with default c++ style indices
    if size(coefficients, 2) < max_num_of_terms_per_poly
        remain_num_of_terms = max_num_of_terms_per_poly - size(coefficients, 2);
        for ti = 1:remain_num_of_terms
            %> define the starting index
            begin_idx = (ti-1)*max_num_of_parts_per_term + (size(coefficients, 2))*max_num_of_parts_per_term;

            %> push 0 to the scalar
            Ht_indices(p, 1 + begin_idx) = 0;

            %> push the c++indexing style for coefficients
            Ht_indices(p, 2 + begin_idx) = size(unique_coeffs, 2);

            %> push the c++ indexing style for variables
            remain_num_of_vars = max_num_of_parts_per_term-2;
            for vi = 1:remain_num_of_vars
                Ht_indices(p, 2 + vi + begin_idx) = numOfVars;
            end
        end
    end 
end

if params.write_Ht_indices
    for i = 1:numOfVars
        for j = 1:max_num_of_terms_per_poly*max_num_of_parts_per_term
            fprintf(outputFileWr_Ht_indices, string(Ht_indices(i, j)));
            fprintf(outputFileWr_Ht_indices, '\t');
        end
        fprintf(outputFileWr_Ht_indices, '\n');
    end
end

% ========================================================================
% > Extract indices of the Jacobian Hx
% ========================================================================
%> Now we can extract indices from the Jacobian Hx
fprintf('Extracting indices of the Jacobian Hx ... \n');
max_num_of_parts_per_Hx_term = max_num_of_parts_per_term - 1;
max_num_of_vars_per_Hx_term = max_num_of_vars_per_term - 1;
Hx_indices = zeros(numOfVars, max_num_of_parts_per_Hx_term*max_num_of_terms_per_Hx_poly);

for p = 1:numOfVars
    for v = 1:numOfVars
        target_poly = jacobian(str2sym(replaced_polys(p)), X(v));
        
        %> isolate variables
        [coefficients, variables] = coeffs(target_poly, X);
        
        %> if the entry is zero, push the default values
        if size(coefficients, 2) == 0
            for ti = 1:max_num_of_terms_per_Hx_poly
                %> define the starting index
                begin_idx = (ti-1)*max_num_of_parts_per_Hx_term;
                
                %> push 0 to the scalar
                Hx_indices(v, 1 + begin_idx) = 0;
                
                %> push the c++indexing style for coefficients
                Hx_indices(v, 2 + begin_idx) = size(unique_coeffs, 2);
                
                %> push the c++ indexing style for variables
                remain_num_of_vars = max_num_of_parts_per_Hx_term-2;
                for vi = 1:remain_num_of_vars
                    Hx_indices(v, 2 + vi + begin_idx) = numOfVars;
                end
            end
            continue;
        end

        %> for each coefficient
        for ci = 1:size(coefficients, 2)
            %> define the starting index
            begin_idx = (ci-1)*max_num_of_parts_per_Hx_term+1;

            %> for each coefficient, isolate scalars
            [scalars, rep_cs] = coeffs(coefficients(1,ci), C);

            %> push the scalar
            Hx_indices(v, begin_idx) = scalars;

            %> push the coefficient index matched with c++
            c_idx = extractAfter(string(rep_cs), "c");
            cpp_c_idx = double(c_idx)-1;
            Hx_indices(v, begin_idx+1) = cpp_c_idx;
            
            %> for the variable, check the power first
            str_var = string(variables(1,ci));
            [~, expanded_vars] = expand_powers_in_poly_term(str_var);
            vars_parts = strsplit(expanded_vars, "*");

            %> push the variable indices matched with c++
            if strcmp(vars_parts, "1")
                remain_num_of_vars = max_num_of_vars_per_Hx_term;
                for ri = 1:remain_num_of_vars
                    Hx_indices(v, begin_idx+size(vars_parts, 2)+ri) = numOfVars;
                end
                continue;
            else
                for vi = 1:size(vars_parts, 2)
                    v_idx = extractAfter(vars_parts(1,vi), "x");
                    cpp_v_idx = double(v_idx)-1;
                    Hx_indices(v, begin_idx+1+vi) = cpp_v_idx;
                end

                if size(vars_parts, 2) < max_num_of_vars_per_Hx_term
                    remain_num_of_vars = max_num_of_vars_per_Hx_term - size(vars_parts, 2);
                    for ri = 1:remain_num_of_vars
                        Hx_indices(v, begin_idx+size(vars_parts, 2)+1+ri) = numOfVars;
                    end
                end
            end 
        end
        
        %> append the rest of the terms with default c++ style indices
        if size(coefficients, 2) < max_num_of_terms_per_Hx_poly
            remain_num_of_terms = max_num_of_terms_per_Hx_poly - size(coefficients, 2);
            for ti = 1:remain_num_of_terms
                %> define the starting index
                begin_idx = (ti-1)*max_num_of_parts_per_Hx_term + (size(coefficients, 2))*max_num_of_parts_per_Hx_term;

                %> push 0 to the scalar
                Hx_indices(v, 1 + begin_idx) = 0;

                %> push the c++indexing style for coefficients
                Hx_indices(v, 2 + begin_idx) = size(unique_coeffs, 2);

                %> push the c++ indexing style for variables
                remain_num_of_vars = max_num_of_parts_per_Hx_term-2;
                for vi = 1:remain_num_of_vars
                    Hx_indices(v, 2 + vi + begin_idx) = numOfVars;
                end
            end
        end
    end
    
    if params.write_Hx_indices
        for i = 1:numOfVars
            for j = 1:max_num_of_terms_per_Hx_poly*max_num_of_parts_per_Hx_term
                fprintf(outputFileWr_Hx_indices, string(Hx_indices(i,j)));
                fprintf(outputFileWr_Hx_indices, '\t');
            end
        end
        fprintf(outputFileWr_Hx_indices, '\n');
    end
end

fprintf("It's Finished!\n");

%> print out the necessary information of the problem
fprintf("\n ------------------------------------------------------- \n");
fprintf("Number of coefficients from parameters: ");
fprintf(string(size(unique_coeffs, 2)));
fprintf('\n');
fprintf("Maximal order of function t: ");
fprintf(string(max_order_of_func_t));
fprintf("\n ------------------------------------------------------- \n");
fprintf("Max number of terms per polynomial: ");
fprintf(string(max_num_of_terms_per_poly));
fprintf('\n');
fprintf("Max number of parts per term: ");
fprintf(string(max_num_of_parts_per_term));
fprintf("\n ------------------------------------------------------- \n");
fprintf("Max number of terms per Hx entry polynomial: ");
fprintf(string(max_num_of_terms_per_Hx_poly));
fprintf('\n');
fprintf("Max number of parts per Hx term polynomial: ");
fprintf(string(max_num_of_parts_per_Hx_term));
fprintf("\n ------------------------------------------------------- \n");

%> close the files
fclose(outputFileWr_P2C_Hx_script);
fclose(outputFileWr_P2C_Ht_script);
fclose(outputFileWr_rep_problem);
fclose(outputFileWr_keep_params_rep_problem);

if params.write_Ht_indices
    fclose(outputFileWr_Ht_indices);
end

if params.write_Hx_indices
    fclose(outputFileWr_Hx_indices);
end
