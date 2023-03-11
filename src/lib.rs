use nih_plug::prelude::*;
use std::io::BufReader;
use std::sync::Arc;
use tract_onnx::prelude::*;

struct OnnxAudioPlugin {
    params: Arc<OnnxAudioPluginParams>,
    input_vec: tract_ndarray::Array4<f32>,
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

#[derive(Params)]
struct OnnxAudioPluginParams {}

impl Default for OnnxAudioPlugin {
    fn default() -> Self {
        // let onnx_model = include_bytes!("../linear.onnx");
        // let onnx_model = include_bytes!("../sin.onnx");
        let onnx_model = include_bytes!("../tanh.onnx");
        let model = onnx()
            // load the model
            .model_for_read(&mut BufReader::new(&onnx_model[..]))
            .unwrap()
            // optimize the model
            .into_optimized()
            .unwrap()
            // make the model runnable and fix its inputs and outputs
            .into_runnable()
            .unwrap();

        Self {
            params: Arc::new(OnnxAudioPluginParams::default()),
            input_vec: tract_ndarray::Array4::<f32>::zeros((1, 1, 1, 1)),
            model,
        }
    }
}

impl Default for OnnxAudioPluginParams {
    fn default() -> Self {
        Self {}
    }
}

impl Plugin for OnnxAudioPlugin {
    const NAME: &'static str = "Onnx Audio Plugin";
    const VENDOR: &'static str = "Akiyuki Okayasu";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "akiyuki.okayasu@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),

            aux_input_ports: &[],
            aux_output_ports: &[],

            // Individual ports and the layout as a whole can be named here. By default these names
            // are generated as needed. This layout will be called 'Stereo', while the other one is
            // given the name 'Mono' based no the number of input and output channels.
            names: PortNames::const_default(),
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Resize buffers and perform other potentially expensive initialization operations here.
        // The `reset()` function is always called right after this function. You can remove this
        // function if you do not need it.
        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                self.input_vec[[0, 0, 0, 0]] = *sample;
                let v = self.input_vec.clone().into_tvalue();
                let result = self.model.run(tvec!(v)).unwrap();
                let output_vec = result[0].to_array_view::<f32>().unwrap();
                *sample = output_vec[[0, 0, 0, 0]];
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for OnnxAudioPlugin {
    const CLAP_ID: &'static str = "com.groundless-electronics.onnx-audio-plugin";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Audio plug-in example using ONNX");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for OnnxAudioPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"onnxaudioplugin ";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Tools,
        Vst3SubCategory::Stereo,
        Vst3SubCategory::Mono,
    ];
}

nih_export_clap!(OnnxAudioPlugin);
nih_export_vst3!(OnnxAudioPlugin);
