package ai.knowly.langtorch.capability.modality.text;

import ai.knowly.langtorch.processor.EmbeddingProcessor;
import ai.knowly.langtorch.processor.Processor;
import ai.knowly.langtorch.prompt.template.PromptTemplate;
import ai.knowly.langtorch.schema.chat.ChatMessage;
import ai.knowly.langtorch.schema.chat.UserMessage;
import ai.knowly.langtorch.schema.embeddings.EmbeddingInput;
import ai.knowly.langtorch.schema.embeddings.EmbeddingOutput;
import ai.knowly.langtorch.schema.io.DomainDocument;
import ai.knowly.langtorch.schema.text.MultiChatMessage;
import ai.knowly.langtorch.store.memory.Memory;
import ai.knowly.langtorch.store.memory.conversation.ConversationMemoryContext;
import ai.knowly.langtorch.store.vectordb.integration.VectorStore;
import ai.knowly.langtorch.store.vectordb.integration.schema.SimilaritySearchQuery;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.FluentLogger;
import com.google.common.util.concurrent.FluentFuture;
import com.google.common.util.concurrent.ListenableFuture;

import javax.inject.Inject;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

//TODO:
// This class is a copy and paste of ChatCompletionLLMCapability.
// I've added a vector store at line 32.
// Our end goal is to build an example in the empty class file 'ChatBotWithVectorStore'
// that can carry out a chat conversation with memory and respect to a vector store
// :/
/** Capability for a chat completion language model with respect to an embeddings vector store. */
public class ChatCompletionWithVectorStoreLLMCapability<I, O>
    implements TextLLMCapabilityWithMemory<
        I, MultiChatMessage, ChatMessage, O, ChatMessage, ConversationMemoryContext> {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  private final Processor<MultiChatMessage, ChatMessage> processor;
  private final EmbeddingProcessor embeddingProcessor;
  private final Parsers<I, MultiChatMessage, ChatMessage, O> parsers;
  private final Memory<ChatMessage, ConversationMemoryContext> memory;
  private final VectorStore vectorStore;
  private boolean verbose;

  @Inject
  public ChatCompletionWithVectorStoreLLMCapability(
          Processor<MultiChatMessage, ChatMessage> processor,
          EmbeddingProcessor embeddingProcessor, Parsers<I, MultiChatMessage, ChatMessage, O> parsers,
          Memory<ChatMessage, ConversationMemoryContext> memory,
          VectorStore vectorStore) {
    this.processor = processor;
    this.embeddingProcessor = embeddingProcessor;
    this.parsers = parsers;
    this.memory = memory;
    this.vectorStore = vectorStore;
    this.verbose = false;
  }

  protected ChatCompletionWithVectorStoreLLMCapability<I, O> withVerboseMode(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public MultiChatMessage preProcess(I inputData) {
    if (inputData instanceof MultiChatMessage) {
      return (MultiChatMessage) inputData;
    }
    return parsers
        .getInputParser()
        .map(parser -> parser.parse(inputData))
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "Input data is not a MultiChatMessage and no input parser is present."));
  }

  @Override
  public O postProcess(ChatMessage outputData) {
    return parsers
        .getOutputParser()
        .map(parser -> parser.parse(outputData))
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "Output data type is not ChatMessage and no output parser is present."));
  }

  @Override
  public Memory<ChatMessage, ConversationMemoryContext> getMemory() {
    return memory;
  }

  @Override
  public O run(I inputData) {
    return postProcess(generateMemorySideEffectResponse(preProcess(inputData)));
  }

  //TODO: not sure if we need this since its going to be duplicated from the copied class? possibly we should extend that class instead?
  private ChatMessage generateMemorySideEffectResponse(MultiChatMessage multiChatMessage) {
    if (verbose) {
      logger.atInfo().log("Memory before processing: %s", memory);
    }

    //TODO:: use parser to create the template maybe?
    String chatHistory = memory.getMemoryContext().get();

    //TODO:: use more safe approach. Maybe return something other than MultiChatMessage from parser?
    String userMessage = multiChatMessage.getMessages().get(0).getContent();

    //TODO:: constant
    String template = "`Given the following conversation and a follow up question," +
            " rephrase the follow up question to be a standalone question. \n" +
            "{{$chatHistory}} \n" +
            "Follow Up question: {{$question}} \n" +
            "Standalone question:`";


    String formattedTemplate = PromptTemplate.builder()
            .setTemplate(template)
            .addAllVariableValuePairs(ImmutableMap.of("$chatHistory", chatHistory, "$question", userMessage))
            .build()
            .format();

    //
    String standaloneQuestion = processor.run(MultiChatMessage.of(UserMessage.of(formattedTemplate))).getContent();
    EmbeddingOutput embeddingOutput = embeddingProcessor.run(EmbeddingInput.builder()
            .setModel("text-embedding-ada-002") //TODO:: in constructor maybe?
            .setInput(Collections.singletonList(standaloneQuestion))
            .build());
    List<Double> vector = embeddingOutput.getValue().get(0).getVector();
    //TODO:: what if similaritySearch could process string input?
    List<DomainDocument> documentsWithScores = vectorStore.similaritySearch(
            SimilaritySearchQuery.builder()
                    .setTopK(5L) //TODO:: in constructor maybe?
                    .setQuery(vector)
                    .build()
    );

    String context = documentsWithScores.stream()
            .map(DomainDocument::getPageContent)
            .collect(Collectors.joining("\n"));

    //TODO:: constant
    String prompt = "`You are a helpful AI assistant. " +
            "Use the following pieces of context to answer the question at the end. " +
            "If you don't know the answer, just say you don't know. " +
            "DO NOT try to make up an answer. If the question is not related to the context, " +
            "politely respond that you are tuned to only answer questions that are related to the context. \n" +
            "Context: \n  {{$context}} \n\n" +
            "Question: \n {{$question}}  \n\n" +
            "Helpful answer in markdown:`";

    String questionWithPrompt = PromptTemplate.builder()
            .setTemplate(prompt)
            .addAllVariableValuePairs(ImmutableMap.of("context", context, "question", standaloneQuestion))
            .build()
            .format();

    ChatMessage response = processor.run(MultiChatMessage.of(UserMessage.of(questionWithPrompt)));

    //TODO::
    memory.add(UserMessage.of(standaloneQuestion));
    memory.add(response);

//    ChatMessage response = processor.run(getMessageWithMemorySideEffect(multiChatMessage));
    // Adding prompt and response.
//    multiChatMessage.getMessages().forEach(memory::add);
//    memory.add(response);
    return response;
  }

  //TODO: ditto
  private MultiChatMessage getMessageWithMemorySideEffect(MultiChatMessage message) {
    // Memory context being empty means that this is the first message in the conversation
    String memoryContext = memory.getMemoryContext().get();
    if (memoryContext.isEmpty()) {
      return message;
    }

    MultiChatMessage updatedMessage =
        message.getMessages().stream()
            .map(
                chatMessage ->
                    new ChatMessage(
                        String.format(
                            "%s%nBelow is my query:%n%s", memoryContext, chatMessage.toString()),
                        chatMessage.getRole(),
                        null,
                        null))
            .collect(MultiChatMessage.toMultiChatMessage());

    if (verbose) {
      logger.atInfo().log("Updated Message with Memory Side Effect: %s", updatedMessage);
    }

    return updatedMessage;
  }

  @Override
  public ListenableFuture<O> runAsync(I inputData) {
    return FluentFuture.from(immediateFuture(inputData)).transform(this::run, directExecutor());
  }
}
