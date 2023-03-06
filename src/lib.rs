use nih_plug::prelude::*;
use std::io::BufReader;
use std::sync::Arc;
use tract_onnx::prelude::*;

// This is a shortened version of the gain example with most comments removed, check out
// https://github.com/robbert-vdh/nih-plug/blob/master/plugins/examples/gain/src/lib.rs to get
// started

struct OnnxAudioPlugin {
    params: Arc<OnnxAudioPluginParams>,
    // input_vec: SmallVec<[TValue; 4]>,
    input_vec: tract_ndarray::Array4<f32>,
    // tensor: Tensor::<f32>,
    // model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>, // model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

#[derive(Params)]
struct OnnxAudioPluginParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    #[id = "gain"]
    pub gain: FloatParam,
}

impl Default for OnnxAudioPlugin {
    fn default() -> Self {
        let onnx_model = include_bytes!("../linear.onnx");
        let model = onnx()
            .model_for_read(&mut BufReader::new(&onnx_model[..]))
            .unwrap()
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec![1, 1, 1, 1]),
            )
            .unwrap()
            .with_output_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec![1, 1, 1, 1]),
            )
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();

        Self {
            params: Arc::new(OnnxAudioPluginParams::default()),
            // input_vec: SmallVec::<[f32; 4]>::new(),
            // input_vec: tract_ndarray::arr1(&[0.0f32]),
            input_vec: tract_ndarray::Array4::<f32>::zeros((1, 1, 1, 1)),
            model: model,
        }
    }
}

impl Default for OnnxAudioPluginParams {
    fn default() -> Self {
        Self {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

impl Plugin for OnnxAudioPlugin {
    const NAME: &'static str = "Onnx Test";
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
        // println!("input_vec: {:?}", self.input_vec);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for channel_samples in buffer.iter_samples() {
            // Smoothing is optionally built into the parameters themselves
            let gain = self.params.gain.smoothed.next();

            for sample in channel_samples {
                // self.input_vec.fill(*sample);
                // self.input_vec[[0, 0, 0, 0]] = *sample;
                let tensor = self.input_vec.clone().into_tensor();
                let result = self.model.run(tvec![tensor.into()]).unwrap();
                let to_show = result[0].to_array_view::<f32>().unwrap();
                // let s = to_show[0];
                println!("result: {to_show}");
                // *sample = s;
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

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for OnnxAudioPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"onnxaudioplugin ";

    // And don't forget to change these categories, see the docstring on `VST3_SUBCATEGORIES` for more
    // information
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(OnnxAudioPlugin);
nih_export_vst3!(OnnxAudioPlugin);
