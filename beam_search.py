import tensorflow as tf
class Hypothesis(object):
    """Defines a hypothesis during beam search."""
    def __init__(self, tokens, prob, state):
        """Hypothesis constructor.
        Args:
        tokens: start tokens for decoding.
        prob: prob of the start tokens, usually 1.
        state: decoder initial states.
        """
        self.tokens = tokens
        self.prob = prob
        self.log_prob = tf.log(prob)
        self.state = state

    @property
    def latest_token(self):
        return self.tokens[-1]

    def Extend(self, token, prob, new_state):
        """Extend the hypothesis with result from latest step.
        Args:
        token: latest token from decoding.
        prob: prob of the latest decoded tokens.
        new_state: decoder output state. Fed to the decoder for next step.
        Returns:
        New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens + [token], self.prob + prob, new_state)

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
        self.decoder = decoder

    def beam_search(self, dec_input, enc_init_state):
        # Replicate the initial states K times for the first step.
        hyps = [Hypothesis([self.start_token], 0.0, enc_init_state)]
        results = []
        steps = 0
        while steps < self.max_steps and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]
            new_states = []
            topk_probs = []
            topk_ids = []
            for idx in range(len(latest_tokens)):
                dec_input = tf.expand_dims([latest_tokens[idx]], 1)
                new_predicts, new_state = self.decoder(dec_input, states[idx])
                _topk_probs, _topk_ids = tf.nn.top_k(new_predicts, self.beam_size)
                topk_probs.append(_topk_probs.numpy().flatten())
                topk_ids.append(_topk_ids.numpy().flatten())
                new_states.append(new_state)
            # Extend each hypothesis.
            # The first step takes the best K results from first hyps. Following
            # steps take the best K results from K*K hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            all_hyps = []
            for i in range(num_beam_source):
                #print("i",i)
                h, ns = hyps[i], new_states[i]
                #print("h",h)
                for j in range(self.beam_size):
                    #print("j",j)
                    all_hyps.append(h.Extend(topk_ids[i][j], topk_probs[i][j], ns))

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for h in self.best_hyps(all_hyps):
                if h.latest_token == self.end_token:
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                    if len(hyps) == self.beam_size or len(results) == self.beam_size:
                        break

            steps += 1

        if steps == self.max_steps:
            results.extend(hyps)

        return self.best_hyps(results)[0]

    def best_hyps(self, hyps):
        return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)