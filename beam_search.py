import tensorflow as tf
class Hypothesis(object):
    """Defines a hypothesis during beam search."""
    def __init__(self, tokens, log_prob, state):
        """Hypothesis constructor.
        Args:
        tokens: start tokens for decoding.
        log_prob: log prob of the start tokens, usually 1.
        state: decoder initial states.
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    @property
    def latest_token(self):
        return self.tokens[-1]

    def Extend(self, token, log_prob, new_state):
        """Extend the hypothesis with result from latest step.
        Args:
        token: latest token from decoding.
        log_prob: log prob of the latest decoded tokens.
        new_state: decoder output state. Fed to the decoder for next step.
        Returns:
        New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob, new_state)

class BeamSearch(object):
    def __init__(self,
        beam_size,
        start_token,
        end_token,
        targ_lang,
        max_steps,
        batch_size,
        decoder):

        self.beam_size = beam_size
        self.start_token = start_token
        self.end_token = end_token
        self.targ_lang = targ_lang
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.decoder = decoder

    def beam_search(self,
            dec_input,
            enc_states,
            lstm=False
        ):
        # Replicate the initial states K times for the first step.
        hyps = [Hypothesis([self.start_token], 0.0, enc_states)] * self.beam_size
        results = []
        steps = 0
        while steps < self.max_steps and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in hyps]
            if steps > 0:
                dec_input = tf.expand_dims(latest_tokens, 1)
                if lstm:
                    enc_states_h, enc_states_c = [], []
                    [enc_states_h.append(h.state[0]) for h in hyps]
                    [enc_states_c.append(h.state[1]) for h in hyps]
                    enc_states = [tf.stack(enc_states_h), tf.stack(enc_states_c)]
                else:
                    enc_states = tf.convert_to_tensor([h.state for h in hyps])

            predictions, new_states = self.decoder(dec_input, enc_states)
            predictions = tf.squeeze(predictions, axis=1)
            topk_probs, topk_ids = tf.nn.top_k(tf.log(predictions), self.beam_size * 2)
            # Extend each hypothesis.
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            all_hyps = []
            for i in range(num_beam_source):
                if lstm:
                    h, ns = hyps[i], [new_states[0][i], new_states[1][i]]
                else:
                    h, ns = hyps[i], new_states[i]

                for j in range(self.beam_size * 2):
                    _h = h.Extend(topk_ids[i, j].numpy(), topk_probs[i, j].numpy(), ns)
                    all_hyps.append(_h)

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for h in self.best_hyps(all_hyps):
                if h.latest_token == self.end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if (len(hyps) == self.beam_size or len(results) == self.beam_size):
                    break
            steps += 1

        if steps == self.max_steps:
            results.extend(hyps)

        return self.best_hyps(results)[0]

    def best_hyps(self, hyps):
        return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)