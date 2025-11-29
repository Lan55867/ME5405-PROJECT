function A = readAlphanumericImage(filename)
    % 此代码基于 Prj_Computer - Examples.pdf (Page 1) [cite: 78-96]
    % 和 ME5405 2025.pdf (Page 1) [cite: 366]
    % 读取一个 64x64 的 .txt 文件，其中包含 0-9 和 A-V 的 32 级灰度
    
    % open the file
    fid = fopen(filename);
    if fid == -1
        error('无法打开文件: %s。\n请确保文件在 MATLAB 路径中。', filename);
    end
    
    % read a char at a time, ignore linefeed and carriage return
    % and put them in a 64x64 matrix
    lf = char(10); % line feed character
    cr = char(13); % carriage return character
    
    % fscanf 读取 64x64 个字符，忽略 lf 和 cr
    try
        A = fscanf(fid, [cr lf '%c'], [64, 64]);
    catch
        % 如果 64x64 失败 (例如 chromo.txt 和 charact1.txt 格式不同)
        fseek(fid, 0, 'bof'); % 重置文件指针
        A = fscanf(fid, '%c');
        % 假设它至少有 64*64=4096 个字符
        if length(A) >= 4096
            A = reshape(A(1:4096), [64, 64]);
        else
            fclose(fid);
            error('文件 %s 不是 64x64 格式', filename);
        end
    end
    
    % close the file handler
    fclose(fid);
    
    % transpose since fscanf returns column vectors
    A = A'; 
    
    % convert letters A-V to their corresponding values (10-31) [cite: 366]
    letters = A >= 'A' & A <= 'V';
    A(letters) = A(letters) - 'A' + 10;
    
    % convert number literals 0-9 to their corresponding values (0-9) [cite: 366]
    digits = A >= '0' & A <= '9';
    A(digits) = A(digits) - '0';
    
    % 将其他所有字符（例如空格, 'v' 等）设为 0
    A(~(digits | letters)) = 0;
    
    % 转换回数值矩阵
    A_out = zeros(size(A));
    A_out(letters) = double(A(letters));
    A_out(digits) = double(A(digits));
    
    % 转换为 uint8 类型的图像矩阵
    A = uint8(A_out);
end