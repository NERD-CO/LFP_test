% Generate Signals with Strong dMI
Fs = 1000;                   % Sampling frequency (Hz)
t = 0:1/Fs:10-1/Fs;          % Time vector (10 seconds)
low_freq = 10;               % Low frequency component (Hz)
high_freq = 100;             % High frequency component (Hz)

% Generate Low Frequency Signal
low_freq_signal = sin(2*pi*low_freq*t);

% Generate High Frequency Signal with Amplitude Modulation
high_amplitude_modulation = 1 + 0.5 * sin(2*pi*low_freq*t); % Amplitude modulated by low_freq signal
high_freq_signal = high_amplitude_modulation .* sin(2*pi*high_freq*t);

% Step 1: Bandpass Filtering (Assuming low_freq_band and high_freq_band are defined)
low_freq_band = [8 12];      % Example low frequency band (Hz)
high_freq_band = [80 120];   % Example high frequency band (Hz)
low_filtered = bandpass(low_freq_signal, low_freq_band, Fs);
high_filtered = bandpass(high_freq_signal, high_freq_band, Fs);

% Step 2: Hilbert Transform to extract phase and amplitude
low_phase = angle(hilbert(low_filtered));
high_amplitude = abs(hilbert(high_filtered)); % Using Hilbert transform for amplitude

% Step 3: Phase-Amplitude Histogram
bin_width = 20;              % Bin width in degrees
phase_bins = -180:bin_width:180;
nBins = length(phase_bins) - 1;
phase_amplitude_hist = zeros(1, nBins);

for i = 1:nBins
    bin_start = phase_bins(i);
    bin_end = phase_bins(i+1);
    idx = low_phase >= deg2rad(bin_start) & low_phase < deg2rad(bin_end);
    phase_amplitude_hist(i) = mean(high_amplitude(idx));
end

% Step 4: Normalize Phase-Amplitude Histogram
p25 = prctile(phase_amplitude_hist, 25);
p75 = prctile(phase_amplitude_hist, 75);
normalized_hist = (phase_amplitude_hist - p25) / (p75 - p25);
normalized_hist = normalized_hist * 2 - 1; % Scale to [-1, 1]

% Step 5: Fit Sinusoid to Normalized Histogram
x = deg2rad(phase_bins(1:end-1) + bin_width/2); % Midpoints of phase bins
sin_func = @(p, x) p(1) * sin(x + p(2)); % Sinusoid function
p0 = [1, 0]; % Initial guess for amplitude and phase
opts = optimset('Display', 'off');
params = lsqcurvefit(sin_func, p0, x, normalized_hist, [0.95, -pi], [1.05, pi], opts);

% Step 6: Calculate dMI
fitted_sinusoid = sin_func(params, x);
error = mean((normalized_hist - fitted_sinusoid).^2);
dMI = max(0, 1 - error);

% Display Results
disp(['Modulation Index (dMI): ', num2str(dMI)]);

% Plot results
figure;
subplot(3,1,1);
plot(t, low_freq_signal);
title('Low Frequency Signal');

subplot(3,1,2);
plot(t, high_freq_signal);
title('High Frequency Signal');

subplot(3,1,3);
plot(rad2deg(x), normalized_hist, 'bo-');
hold on;
plot(rad2deg(x), fitted_sinusoid, 'r-');
title('Phase-Amplitude Histogram with Sinusoidal Fit');
xlabel('Phase (degrees)');
ylabel('Normalized Amplitude');
legend('Normalized Histogram', 'Fitted Sinusoid');

%%
% Sample Data Generation Parameters
Fs = 1000;           % Sampling frequency (Hz)
t = 0:1/Fs:10-1/Fs;  % Time vector (10 seconds)

% Frequency ranges
low_freqs = 1:1:20;  % Low frequency range for phase (Hz)
high_freqs = 30:10:160; % High frequency range for amplitude (Hz)

% Number of trials
nTrials = 100;

% Preallocate space for MI and t-statistics
MI = zeros(length(high_freqs), length(low_freqs));
t_stat = zeros(length(high_freqs), length(low_freqs));

% Generate signals and calculate MI
for lf = 1:length(low_freqs)
    for hf = 1:length(high_freqs)
        low_freq = low_freqs(lf);
        high_freq = high_freqs(hf);
        
        % Generate synthetic data for each trial
        all_MI = zeros(1, nTrials);
        for trial = 1:nTrials
            % Generate low frequency signal
            low_freq_signal = sin(2*pi*low_freq*t);
            
            % Generate high frequency signal with amplitude modulation
            high_amplitude_modulation = 1 + 0.5 * sin(2*pi*low_freq*t);
            high_freq_signal = high_amplitude_modulation .* sin(2*pi*high_freq*t);
            
            % Bandpass filtering with valid bounds
            low_freq_band = [max(0.1, low_freq-1), low_freq+1]; % Ensure positive frequency bounds
            high_freq_band = [high_freq-10, high_freq+10];
            low_filtered = bandpass(low_freq_signal, low_freq_band, Fs);
            high_filtered = bandpass(high_freq_signal, high_freq_band, Fs);
            
            % Hilbert Transform to extract phase and amplitude
            low_phase = angle(hilbert(low_filtered));
            high_amplitude = abs(hilbert(high_filtered));
            
            % Phase-Amplitude Histogram
            bin_width = 20;
            phase_bins = -180:bin_width:180;
            nBins = length(phase_bins) - 1;
            phase_amplitude_hist = zeros(1, nBins);
            
            for i = 1:nBins
                bin_start = phase_bins(i);
                bin_end = phase_bins(i+1);
                idx = low_phase >= deg2rad(bin_start) & low_phase < deg2rad(bin_end);
                phase_amplitude_hist(i) = mean(high_amplitude(idx));
            end
            
            % Normalize Phase-Amplitude Histogram
            p25 = prctile(phase_amplitude_hist, 25);
            p75 = prctile(phase_amplitude_hist, 75);
            normalized_hist = (phase_amplitude_hist - p25) / (p75 - p25);
            normalized_hist = normalized_hist * 2 - 1; % Scale to [-1, 1]
            
            % Fit Sinusoid to Normalized Histogram
            x = deg2rad(phase_bins(1:end-1) + bin_width/2); % Midpoints of phase bins
            sin_func = @(p, x) p(1) * sin(x + p(2)); % Sinusoid function
            p0 = [1, 0]; % Initial guess for amplitude and phase
            opts = optimset('Display', 'off');
            params = lsqcurvefit(sin_func, p0, x, normalized_hist, [0.95, -pi], [1.05, pi], opts);
            
            % Calculate dMI
            fitted_sinusoid = sin_func(params, x);
            error = mean((normalized_hist - fitted_sinusoid).^2);
            all_MI(trial) = max(0, 1 - error);
        end
        
        % Calculate average MI
        MI(hf, lf) = mean(all_MI);
        
        % Calculate t-statistic
        [~, ~, ~, stats] = ttest(all_MI);
        t_stat(hf, lf) = stats.tstat;
    end
end

% Plot MI heatmap
figure;
subplot(1,2,1);
imagesc(low_freqs, high_freqs, MI);
colorbar;
xlabel('Frequency for Phase (Hz)');
ylabel('Frequency for Amplitude (Hz)');
title('Average PAC (MI)');

% Plot t-statistics heatmap
subplot(1,2,2);
imagesc(low_freqs, high_freqs, t_stat);
colorbar;
xlabel('Frequency for Phase (Hz)');
ylabel('Frequency for Amplitude (Hz)');
title('t-statistics');
